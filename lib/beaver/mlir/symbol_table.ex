defmodule Beaver.MLIR.SymbolTable do
  use Kinda.ResourceKind, forward_module: Beaver.Native

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
  Returns the name of the attribute used for symbol visibility as atom.

  ## Examples
    iex> MLIR.SymbolTable.visibility_attribute_name()
    :sym_visibility
  """
  def visibility_attribute_name() do
    mlirSymbolTableGetVisibilityAttributeName() |> to_string() |> String.to_atom()
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
