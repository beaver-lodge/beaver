defmodule Beaver.MLIR.SymbolTable do
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  @moduledoc """
  This module provides utilities for MLIR symbol table.
  """
  alias Beaver.MLIR
  import MLIR.CAPI

  @doc """
  Creates a new symbol table and attaches it to the given operation.
  The operation is expected to have the `SymbolTable` trait.
  """
  defdelegate create(operation), to: MLIR.CAPI, as: :mlirSymbolTableCreate

  @doc "Destroys the given symbol table."
  defdelegate destroy(symbol_table), to: MLIR.CAPI, as: :mlirSymbolTableDestroy

  @doc "Erases a symbol from the given symbol table."
  defdelegate erase(symbol_table, operation), to: MLIR.CAPI, as: :mlirSymbolTableErase

  @doc """
  Returns the name of the attribute used for symbol names as atom.

  ## Examples
    iex> MLIR.SymbolTable.attribute_name()
    :sym_name
  """

  def attribute_name() do
    mlirSymbolTableGetSymbolAttributeName() |> to_string() |> String.to_atom()
  end

  @doc """
  Returns the attribute name used by the default symbol visibility
  implementation as an atom.

  A symbol operation may implement visibility without storing this attribute.
  Use `visibility/1` and `set_visibility/2` to access its semantic visibility.

  ## Examples
    iex> MLIR.SymbolTable.default_visibility_attribute_name()
    :sym_visibility
  """
  def default_visibility_attribute_name() do
    beaverSymbolTableGetDefaultVisibilityAttributeName() |> to_string() |> String.to_atom()
  end

  @type visibility :: :public | :private | :nested

  @doc "Returns a symbol operation's semantic visibility."
  @spec visibility(MLIR.Operation.t()) :: visibility()
  def visibility(operation) do
    operation
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
  def set_visibility(operation, visibility) do
    encoded =
      case visibility do
        :public -> 0
        :private -> 1
        :nested -> 2
        other -> raise ArgumentError, "invalid symbol visibility: #{inspect(other)}"
      end

    beaverSymbolTableSetSymbolVisibility(operation, encoded)
  end

  @doc "Inserts an operation into a symbol table."
  defdelegate insert(symbol_table, operation), to: MLIR.CAPI, as: :mlirSymbolTableInsert

  @doc "Looks up a symbol with the given name in the symbol table."
  def lookup(symbol_table, name) do
    mlirSymbolTableLookup(symbol_table, MLIR.StringRef.create(name))
  end

  @doc "Replaces all uses of a symbol with another symbol."
  defdelegate replace_all_symbol_uses(old_symbol, new_symbol, from),
    to: MLIR.CAPI,
    as: :mlirSymbolTableReplaceAllSymbolUses
end
