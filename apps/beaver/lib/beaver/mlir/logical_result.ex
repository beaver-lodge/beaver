defmodule Beaver.MLIR.LogicalResult do
  alias Beaver.MLIR

  def success?(result) do
    result
    |> MLIR.CAPI.beaverLogicalResultIsSuccess()
    |> MLIR.CAPI.to_term()
  end
end
