defmodule Beaver.MLIR.LogicalResult do
  alias Beaver.MLIR

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  def success?(result) do
    result
    |> MLIR.CAPI.beaverLogicalResultIsSuccess()
    |> Beaver.Native.to_term()
  end
end
