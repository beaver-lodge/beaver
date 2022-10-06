defmodule Beaver.MLIR.LogicalResult do
  alias Beaver.MLIR

  def success?(result) do
    result
    |> MLIR.CAPI.beaverLogicalResultIsSuccess()
    |> Beaver.Native.to_term()
  end
end
