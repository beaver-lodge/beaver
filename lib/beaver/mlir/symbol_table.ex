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

    type =
      case function_type do
        s when is_binary(s) -> Type.get(s, ctx: ctx)
        %Beaver.Deferred{} = d -> Beaver.Deferred.resolve(d, ctx)
        %Type{} = t -> t
      end

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
      Operation.with_symbol_table(src, fn src_st -> import_worklist!(dst_st, src_st, seeds, MapSet.new(), opts) end)
    end)

    target
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

  defp import_worklist!(_dst, _src, [], _visited, _opts), do: :ok

  defp import_worklist!(dst, src, [name | rest], visited, opts) do
    deps = if MapSet.member?(visited, name), do: [], else: import_symbol!(dst, lookup(src, name), name, opts)
    import_worklist!(dst, src, deps ++ rest, MapSet.put(visited, name), opts)
  end

  defp import_symbol!(table, src, name, opts) do
    if MLIR.null?(src), do: [], else: merge_symbol!(table, src, lookup(table, name), name, opts)
  end

  defp verify_if_requested!(op, name, type, opts) do
    if opts[:verify_type] && not MLIR.null?(op), do: verify_existing_type!(op, name, type)
  end

  defp merge_symbol!(table, src, dst, name, opts) do
    missing? = MLIR.null?(dst)
    definition? = symbol_definition?(src)

    if type = src["function_type"], do: verify_if_requested!(dst, name, Attribute.value(type), opts)

    if missing? or (definition? and not symbol_definition?(dst)) do
      unless missing?, do: erase(table, dst)
      clone = Operation.clone(src)
      visibility = Keyword.get(opts, :visibility, :private)
      if definition? and visibility != :preserve, do: set_visibility(clone, visibility)
      insert(table, clone)
      if definition?, do: referenced_symbols(src), else: []
    else
      []
    end
  end

  defp symbol_definition?(op) do
    regions = Enum.to_list(Walker.regions(op))
    regions == [] or Enum.any?(regions, &(Region.first_block(&1) != nil)) or Operation.attribute(op, "value") != nil
  end
end
