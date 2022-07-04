defmodule Beaver.MLIR.Sigils do
  alias Beaver.MLIR

  @doc """
  create a module from global MLIR context or process MLIR context if registered.
  """
  def sigil_m(string, []) do
    MLIR.Module.create(string)
  end

  def sigil_a(string, []), do: MLIR.Attribute.get(string)

  def sigil_a(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Attribute.get("#{string} : #{modifier}")
  end

  def sigil_t(string, []), do: MLIR.Type.get(string)

  def sigil_t(string, modifier) do
    modifier = modifier |> List.to_string()
    MLIR.Type.get("#{modifier}<#{string}>")
  end
end
