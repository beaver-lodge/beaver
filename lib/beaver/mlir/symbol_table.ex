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
end
