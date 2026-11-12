alias Beaver.MLIR.CAPI

defmodule Beaver.MLIR.SymbolTable do
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Beaver.Native
  @moduledoc "MLIR symbol lookup, declarations, visibility, and transitive definition imports."
  alias Beaver.{Changeset, MLIR, Native, Walker}
  alias Beaver.MLIR.{Attribute, Operation, Region, StringRef, Type}
  import CAPI
  @type visibility :: :public | :private | :nested
  @visibilities [:public, :private, :nested]

  @doc "Creates a symbol table for an operation with the SymbolTable trait."
  defdelegate create(operation), to: CAPI, as: :mlirSymbolTableCreate
  @doc "Destroys the given symbol table."
  defdelegate destroy(symbol_table), to: CAPI, as: :mlirSymbolTableDestroy
  @doc "Erases a symbol from the given symbol table."
  defdelegate erase(symbol_table, operation), to: CAPI, as: :mlirSymbolTableErase
  @doc "Inserts an operation into a symbol table."
  defdelegate insert(symbol_table, operation), to: CAPI, as: :mlirSymbolTableInsert
  @doc "Replaces all uses of a symbol with another symbol."
  defdelegate replace_all_symbol_uses(old_symbol, new_symbol, from), to: CAPI, as: :mlirSymbolTableReplaceAllSymbolUses

  @doc """
  Returns the symbol-name attribute name as an atom.

      iex> MLIR.SymbolTable.attribute_name()
      :sym_name
  """
  def attribute_name, do: mlirSymbolTableGetSymbolAttributeName() |> to_string() |> String.to_atom()

  @doc """
  Returns the default visibility attribute name; use visibility/1 for semantic visibility.

      iex> MLIR.SymbolTable.default_visibility_attribute_name()
      :sym_visibility
  """
  def default_visibility_attribute_name,
    do: beaverSymbolTableGetDefaultVisibilityAttributeName() |> to_string() |> String.to_atom()

  @doc "Returns a symbol operation's semantic visibility."
  @spec visibility(Operation.t()) :: visibility()
  def visibility(op), do: Enum.fetch!(@visibilities, op |> beaverSymbolTableGetSymbolVisibility() |> Native.to_term())
  @doc "Sets a symbol operation's semantic visibility."
  @spec set_visibility(Operation.t(), visibility()) :: :ok
  def set_visibility(op, visibility) do
    encoded = Enum.find_index(@visibilities, &(&1 == visibility))
    if is_nil(encoded), do: raise(ArgumentError, "invalid symbol visibility: #{inspect(visibility)}")
    beaverSymbolTableSetSymbolVisibility(op, encoded)
  end

  @doc "Looks up a symbol with the given name in the symbol table."
  def lookup(symbol_table, name), do: mlirSymbolTableLookup(symbol_table, StringRef.create(name))

  @doc """
  Finds or creates a declaration in the enclosing symbol table. Options: `:op` (`"func.func"`),
  `:visibility` (`:private`, applied to func.func), `:location`, and `:verify_type` (false).
  With `:verify_type`, an incompatible existing function_type raises ArgumentError.
  """
  @spec ensure_declaration!(
          MLIR.Module.t() | Operation.t(),
          String.Chars.t(),
          Type.t() | Beaver.Deferred.t() | String.t(),
          keyword()
        ) :: Operation.t()
  def ensure_declaration!(module, name, function_type, opts \\ []) do
    root = symbol_root(module)
    ctx = MLIR.context(root)
    name = clean_name(name)

    type = resolve_type!(function_type, ctx)

    Operation.with_symbol_table(root, fn st ->
      op = lookup(st, name)

      if MLIR.null?(op) do
        insert_declaration(st, root, name, type, opts)
      else
        verify_if_requested!(op, name, type, opts)
        op
      end
    end)
  end

  @doc "Returns context-independent `{input_types, result_types}` for immediate func.func symbols."
  def function_signatures(module) do
    for op <- Walker.find_ops(module, ["func.func"], nested: false), into: %{} do
      type = Attribute.value(op["function_type"])

      inputs =
        for i <- 0..(MLIR.FunctionType.num_inputs(type) - 1)//1, do: MLIR.to_string(MLIR.FunctionType.input(type, i))

      results =
        for i <- 0..(MLIR.FunctionType.num_results(type) - 1)//1, do: MLIR.to_string(MLIR.FunctionType.result(type, i))

      {Attribute.value(op["sym_name"]), {inputs, results}}
    end
  end

  @doc "Checks all existing declarations against signatures without changing the module. Missing symbols are allowed."
  def check_declarations!(module, signatures), do: declarations(module, signatures, [], false)

  @doc "Checks every signature before inserting missing declarations; returns module. Accepts types or textual type pairs."
  def ensure_declarations!(module, signatures, opts \\ []), do: declarations(module, signatures, opts, true)

  defp declarations(module, signatures, opts, insert?) do
    root = symbol_root(module)
    ctx = MLIR.context(root)
    types = Enum.map(signatures, fn {name, type} -> {clean_name(name), resolve_type!(type, ctx)} end)

    check_duplicate_types!(types)

    Operation.with_symbol_table(root, fn st ->
      for {name, type} <- types, op = lookup(st, name), not MLIR.null?(op), do: verify_existing_type!(op, name, type)

      if insert?, do: insert_declarations(st, root, types, opts)
    end)

    module
  end

  defp check_duplicate_types!(types) do
    Enum.reduce(types, %{}, fn {name, type}, acc ->
      if acc[name] && not MLIR.equal?(acc[name], type), do: raise(ArgumentError, "conflicting signatures for @#{name}")
      Map.put(acc, name, type)
    end)
  end

  defp insert_declarations(st, root, types, opts) do
    for {name, type} <- types, MLIR.null?(lookup(st, name)), do: insert_declaration(st, root, name, type, opts)
  end

  defp resolve_type!({inputs, results}, ctx) when is_list(inputs) and is_list(results),
    do: Type.function(Enum.map(inputs, &Type.get(&1, ctx: ctx)), Enum.map(results, &Type.get(&1, ctx: ctx)), ctx: ctx)

  defp resolve_type!(type, ctx) when is_binary(type), do: Type.get(type, ctx: ctx)
  defp resolve_type!(%Beaver.Deferred{} = type, ctx), do: Beaver.Deferred.resolve(type, ctx)

  defp resolve_type!(%Type{} = type, ctx) do
    unless MLIR.equal?(MLIR.context(type), ctx), do: raise(ArgumentError, "declaration type belongs to another context")
    type
  end

  defp insert_declaration(st, root, name, type, opts) do
    ctx = MLIR.context(root)
    op_name = Keyword.get(opts, :op, "func.func")
    loc = Keyword.get_lazy(opts, :location, fn -> Operation.location(root) end)
    attrs = [sym_name: Attribute.string(name, ctx: ctx), function_type: Attribute.type(type)]

    attrs =
      if op_name == "func.func",
        do: attrs ++ [sym_visibility: Attribute.string(to_string(Keyword.get(opts, :visibility, :private)), ctx: ctx)],
        else: attrs

    op =
      %Changeset{name: op_name, location: loc, context: ctx}
      |> Changeset.add_argument(attrs)
      |> Changeset.add_argument(Region.create())
      |> Operation.create()

    insert(st, op)
    op
  end

  defp verify_existing_type!(op, name, type) do
    actual = op["function_type"]

    unless actual && not MLIR.null?(actual) && MLIR.equal?(actual, Attribute.type(type)) do
      raise ArgumentError,
            "symbol @#{name} already declared with incompatible function_type #{actual && MLIR.to_string(actual)}, expected #{MLIR.to_string(type)}"
    end
  end

  @doc """
  Clones selected symbols and transitive dependencies from source into target (same context), returning target.
  Replaces bodyless forward declarations with definitions; preserves existing definitions and external declarations.
  Options: `:symbols` (`:referenced` or a list of names), `:visibility` (`:private`, `:public`, `:nested`, `:preserve`),
  and `:verify_type` (false), which checks existing function_type compatibility before replacing a symbol.
  """
  @spec import_symbols!(MLIR.Module.t() | Operation.t(), MLIR.Module.t() | Operation.t(), keyword()) ::
          MLIR.Module.t() | Operation.t()
  def import_symbols!(target, source, opts \\ []) when is_list(opts) do
    dst = symbol_root(target)
    src = symbol_root(source)

    unless MLIR.equal?(MLIR.context(dst), MLIR.context(src)),
      do: raise(ArgumentError, "import_symbols! requires target and source to share the same MLIR.Context")

    seeds = initial_symbols(dst, Keyword.get(opts, :symbols, :referenced))

    Operation.with_symbol_table(dst, fn dst_st ->
      Operation.with_symbol_table(src, fn src_st ->
        imports = plan_imports!(dst_st, src_st, seeds, MapSet.new(), opts, [])
        import_planned!(dst_st, imports, opts)
      end)
    end)

    target
  end

  defp import_planned!(table, imports, opts) do
    for {name, op} <- imports, do: clone_symbol!(table, op, lookup(table, name), opts)
  end

  @doc "Returns unique flat symbol names referenced by operation attributes inside module_or_op."
  @spec referenced_symbols(MLIR.Module.t() | Operation.t()) :: [String.t()]
  def referenced_symbols(module_or_op) do
    root = Operation.from_module(module_or_op)
    Beaver.Execution.run(MLIR.context(root), :referenced_symbols, {root.ref}, strategy: :auto)
  end

  defp clean_name(name), do: name |> to_string() |> String.trim_leading("@")
  defp symbol_root(module), do: then(Operation.from_module(module), &(Operation.enclosing_symbol_table(&1) || &1))
  defp initial_symbols(root, :referenced), do: referenced_symbols(root)
  defp initial_symbols(_root, list) when is_list(list), do: Enum.map(list, &clean_name/1)

  defp initial_symbols(_root, other),
    do: raise(ArgumentError, "invalid :symbols option for import_symbols!: #{inspect(other)}")

  defp plan_imports!(_dst, _src, [], _visited, _opts, imports), do: Enum.reverse(imports)

  defp plan_imports!(dst, src, [name | rest], visited, opts, imports) do
    op = lookup(src, name)

    {deps, imports} =
      if MapSet.member?(visited, name) or MLIR.null?(op),
        do: {[], imports},
        else: plan_symbol!(lookup(dst, name), op, name, opts, imports)

    plan_imports!(dst, src, deps ++ rest, MapSet.put(visited, name), opts, imports)
  end

  defp plan_symbol!(dst, src, name, opts, imports) do
    if type = src["function_type"], do: verify_if_requested!(dst, name, Attribute.value(type), opts)

    if MLIR.null?(dst) or (symbol_definition?(src) and not symbol_definition?(dst)) do
      deps = if symbol_definition?(src), do: referenced_symbols(src), else: []
      {deps, [{name, src} | imports]}
    else
      {[], imports}
    end
  end

  defp verify_if_requested!(op, name, type, opts) do
    if opts[:verify_type] && not MLIR.null?(op), do: verify_existing_type!(op, name, type)
  end

  defp clone_symbol!(table, src, dst, opts) do
    unless MLIR.null?(dst), do: erase(table, dst)
    clone = Operation.clone(src)
    visibility = Keyword.get(opts, :visibility, :private)
    if symbol_definition?(src) and visibility != :preserve, do: set_visibility(clone, visibility)
    insert(table, clone)
  end

  defp symbol_definition?(op) do
    regions = Enum.to_list(Walker.regions(op))
    regions == [] or Enum.any?(regions, &(Region.first_block(&1) != nil)) or Operation.attribute(op, "value") != nil
  end
end
