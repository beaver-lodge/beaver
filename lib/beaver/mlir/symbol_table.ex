alias Beaver.MLIR.CAPI

defmodule Beaver.MLIR.SymbolTable do
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Beaver.Native

  @moduledoc "This module provides utilities for MLIR symbol table."
  alias Beaver.MLIR
  import CAPI

  @doc """
  Creates a new symbol table and attaches it to the given operation. The operation is expected to have the
  `SymbolTable` trait.
  """
  defdelegate create(operation), to: CAPI, as: :mlirSymbolTableCreate

  @doc "Destroys the given symbol table."
  defdelegate destroy(symbol_table), to: CAPI, as: :mlirSymbolTableDestroy

  @doc "Erases a symbol from the given symbol table."
  defdelegate erase(symbol_table, operation), to: CAPI, as: :mlirSymbolTableErase

  @doc """
  Returns the name of the attribute used for symbol names as atom.

  ## Examples
    iex> MLIR.SymbolTable.attribute_name()
    :sym_name
  """

  def attribute_name(), do: mlirSymbolTableGetSymbolAttributeName() |> to_string() |> String.to_atom()

  @doc """
  Returns the attribute name used by the default symbol visibility
  implementation as an atom.

  A symbol operation may implement visibility without storing this attribute.
  Use `visibility/1` and `set_visibility/2` to access its semantic visibility.

  ## Examples
    iex> MLIR.SymbolTable.default_visibility_attribute_name()
    :sym_visibility
  """
  def default_visibility_attribute_name(),
    do: beaverSymbolTableGetDefaultVisibilityAttributeName() |> to_string() |> String.to_atom()

  @type visibility :: :public | :private | :nested

  @doc "Returns a symbol operation's semantic visibility."
  @spec visibility(MLIR.Operation.t()) :: visibility()
  def visibility(op) do
    op
    |> beaverSymbolTableGetSymbolVisibility()
    |> Beaver.Native.to_term()
    |> case do
      0 -> :public
      1 -> :private
      2 -> :nested
    end
  end

  @doc "Sets a symbol operation's semantic visibility."
  @spec set_visibility(MLIR.Operation.t(), visibility()) :: :ok
  def set_visibility(op, visibility) do
    encoded =
      case visibility do
        :public -> 0
        :private -> 1
        :nested -> 2
        other -> raise ArgumentError, "invalid symbol visibility: #{inspect(other)}"
      end

    beaverSymbolTableSetSymbolVisibility(op, encoded)
  end

  @doc "Inserts an operation into a symbol table."
  defdelegate insert(symbol_table, operation), to: CAPI, as: :mlirSymbolTableInsert

  @doc "Looks up a symbol with the given name in the symbol table."
  def lookup(symbol_table, name), do: mlirSymbolTableLookup(symbol_table, MLIR.StringRef.create(name))

  @doc "Replaces all uses of a symbol with another symbol."
  defdelegate replace_all_symbol_uses(old_symbol, new_symbol, from), to: CAPI, as: :mlirSymbolTableReplaceAllSymbolUses

  @doc """
  Looks up or creates an external function/symbol declaration in `module_or_op`'s enclosing symbol table.

  Options:
    * `:op` - declaration operation name (default `"func.func"`).
    * `:visibility` - `:private` (default), `:public`, or `:nested` (applied to `"func.func"`).
    * `:location` - explicit `MLIR.Location` for a newly created declaration.
    * `:verify_type` - when `true`, verifies that an already existing symbol's `function_type`
      attribute matches `function_type` and raises `ArgumentError` on mismatch.
  """
  @spec ensure_declaration!(
          MLIR.Module.t() | MLIR.Operation.t(),
          String.Chars.t(),
          MLIR.Type.t() | Beaver.Deferred.t() | String.t(),
          keyword()
        ) :: MLIR.Operation.t()
  def ensure_declaration!(module_or_op, sym_name, function_type, opts \\ []) do
    op = MLIR.Operation.from_module(module_or_op)
    root_op = MLIR.Operation.enclosing_symbol_table(op) || op
    ctx = MLIR.context(root_op)
    clean_name = sym_name |> to_string() |> String.trim_leading("@")

    type =
      case function_type do
        s when is_binary(s) -> MLIR.Type.get(s, ctx: ctx)
        %Beaver.Deferred{} = d -> Beaver.Deferred.resolve(d, ctx)
        %MLIR.Type{} = t -> t
      end

    Beaver.Native.with_resource(create(root_op), &destroy/1, fn sym_table ->
      lookup_or_insert_declaration!(sym_table, root_op, clean_name, type, opts)
    end)
  end

  defp lookup_or_insert_declaration!(sym_table, root_op, clean_name, type, opts) do
    existing = lookup(sym_table, clean_name)

    if MLIR.null?(existing) do
      insert_declaration(sym_table, root_op, clean_name, type, opts)
    else
      if Keyword.get(opts, :verify_type, false), do: verify_existing_type!(existing, clean_name, type)
      existing
    end
  end

  defp verify_existing_type!(existing, clean_name, expected_type) do
    expected_attr = MLIR.Attribute.type(expected_type)
    actual_attr = existing["function_type"]

    if is_nil(actual_attr) or MLIR.null?(actual_attr) or not MLIR.equal?(actual_attr, expected_attr) do
      raise ArgumentError,
            "symbol @#{clean_name} already declared with incompatible function_type #{actual_attr && MLIR.to_string(actual_attr)}, expected #{MLIR.to_string(expected_type)}"
    end
  end

  defp insert_declaration(sym_table, root_op, clean_name, type, opts) do
    ctx = MLIR.context(root_op)
    op_name = Keyword.get(opts, :op, "func.func")
    loc = Keyword.get_lazy(opts, :location, fn -> MLIR.Operation.location(root_op) end)
    base_attrs = [sym_name: MLIR.Attribute.string(clean_name, ctx: ctx), function_type: MLIR.Attribute.type(type)]
    vis = Keyword.get(opts, :visibility, :private) |> to_string()

    attrs =
      if op_name == "func.func",
        do: Keyword.put(base_attrs, :sym_visibility, MLIR.Attribute.string(vis, ctx: ctx)),
        else: base_attrs

    decl =
      %Beaver.Changeset{name: op_name, location: loc, context: ctx}
      |> Beaver.Changeset.add_argument(attrs)
      |> Beaver.Changeset.add_argument(MLIR.Region.create())
      |> MLIR.Operation.create()

    insert(sym_table, decl)
    decl
  end

  @doc """
  Selectively clones referenced (or explicitly requested) symbol definitions and their transitive symbol
  dependencies from `source` into `target`, returning `target`.

  If `target` contains a bodyless forward declaration for a symbol that `source` defines with a body, the forward
  declaration is replaced with the cloned definition. Existing definitions in `target` are preserved.

  Options:
    * `:symbols` - `:referenced` (default, scans `target` for symbol references) or a list of symbol names.
    * `:visibility` - `:private` (default), `:public`, `:nested`, or `:preserve`.
    * `:verify_type` - when `true`, verifies `function_type` compatibility against existing declarations/definitions.
  """
  @spec import_symbols!(
          MLIR.Module.t() | MLIR.Operation.t(),
          MLIR.Module.t() | MLIR.Operation.t(),
          keyword()
        ) :: MLIR.Module.t() | MLIR.Operation.t()
  def import_symbols!(target, source, opts \\ []) when is_list(opts) do
    dst_root = symbol_root!(target)
    src_root = symbol_root!(source)

    unless MLIR.equal?(MLIR.context(dst_root), MLIR.context(src_root)) do
      raise ArgumentError, "import_symbols! requires target and source to share the same MLIR.Context"
    end

    seeds = initial_import_symbols!(dst_root, Keyword.get(opts, :symbols, :referenced))
    vis = Keyword.get(opts, :visibility, :private)
    verify? = Keyword.get(opts, :verify_type, false)

    Beaver.Native.with_resource(create(dst_root), &destroy/1, fn dst_st ->
      Beaver.Native.with_resource(create(src_root), &destroy/1, fn src_st ->
        import_worklist!(dst_st, src_st, seeds, MapSet.new(), vis, verify?)
      end)
    end)

    target
  end

  @doc "Returns all unique flat symbol names referenced by attributes of operations inside `module_or_op`."
  @spec referenced_symbols(MLIR.Module.t() | MLIR.Operation.t()) :: [String.t()]
  def referenced_symbols(module_or_op) do
    root = MLIR.Operation.from_module(module_or_op)
    Beaver.Execution.run(MLIR.context(root), :referenced_symbols, {root.ref}, strategy: :auto)
  end

  defp symbol_root!(module_or_op) do
    op = MLIR.Operation.from_module(module_or_op)
    MLIR.Operation.enclosing_symbol_table(op) || op
  end

  defp initial_import_symbols!(dst_root, :referenced), do: referenced_symbols(dst_root)

  defp initial_import_symbols!(_dst_root, list) when is_list(list),
    do: Enum.map(list, &(to_string(&1) |> String.trim_leading("@")))

  defp initial_import_symbols!(_dst_root, other),
    do: raise(ArgumentError, "invalid :symbols option for import_symbols!: #{inspect(other)}")

  defp import_worklist!(_dst_st, _src_st, [], _visited, _vis, _verify?), do: :ok

  defp import_worklist!(dst_st, src_st, [sym | rest], visited, vis, verify?) do
    if MapSet.member?(visited, sym) do
      import_worklist!(dst_st, src_st, rest, visited, vis, verify?)
    else
      deps = maybe_import_symbol!(dst_st, src_st, sym, vis, verify?)
      import_worklist!(dst_st, src_st, deps ++ rest, MapSet.put(visited, sym), vis, verify?)
    end
  end

  defp maybe_import_symbol!(dst_st, src_st, sym, vis, verify?) do
    src = lookup(src_st, sym)

    if MLIR.null?(src) do
      []
    else
      merge_symbol!(dst_st, sym, src, lookup(dst_st, sym), vis, verify?)
    end
  end

  defp merge_symbol!(dst_st, sym, src, dst, vis, verify?) do
    src_def? = symbol_definition?(src)

    cond do
      MLIR.null?(dst) ->
        clone_symbol_into!(dst_st, src, src_def?, vis)
        if src_def?, do: referenced_symbols(src), else: []

      symbol_definition?(dst) ->
        if verify?, do: verify_compatible_symbol!(dst, sym, src)
        []

      src_def? ->
        if verify?, do: verify_compatible_symbol!(dst, sym, src)
        erase(dst_st, dst)
        clone_symbol_into!(dst_st, src, true, vis)
        referenced_symbols(src)

      true ->
        if verify?, do: verify_compatible_symbol!(dst, sym, src)
        []
    end
  end

  defp clone_symbol_into!(dst_st, src, true, vis) when vis != :preserve do
    cloned = MLIR.Operation.clone(src)
    set_visibility(cloned, vis)
    insert(dst_st, cloned)
  end

  defp clone_symbol_into!(dst_st, src, _src_def?, _vis) do
    insert(dst_st, MLIR.Operation.clone(src))
  end

  defp symbol_definition?(%MLIR.Operation{} = op) do
    regions = op |> Beaver.Walker.regions() |> Enum.to_list()

    regions == [] or Enum.any?(regions, &(MLIR.Region.first_block(&1) != nil)) or
      MLIR.Operation.attribute(op, "value") != nil
  end

  defp verify_compatible_symbol!(dst, sym, src) do
    case src["function_type"] do
      %MLIR.Attribute{} = src_type_attr ->
        verify_existing_type!(dst, sym, MLIR.Attribute.value(src_type_attr))

      nil ->
        :ok
    end
  end
end
