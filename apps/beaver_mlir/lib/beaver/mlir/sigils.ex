defmodule Beaver.MLIR.Sigils do
  alias Beaver.MLIR

  @doc """
  create a module from global MLIR context or process MLIR context if registered.
  """
  def sigil_m(string, []) do
    MLIR.Module.create(string)
  end

  def sigil_a(string, []), do: String.upcase(string)
  def sigil_t(string, []), do: MLIR.Type.get(string)
end
