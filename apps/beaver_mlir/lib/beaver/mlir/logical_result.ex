defmodule Beaver.MLIR.LogicalResult do
  alias Beaver.MLIR

  def success?(result) do
    result
    |> Exotic.Value.fetch(MLIR.CAPI.MlirLogicalResult, :value)
    |> Exotic.Value.extract() != 0
  end
end
