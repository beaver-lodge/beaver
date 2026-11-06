defmodule Beaver.MLIR.Transform do
  @moduledoc "Standard MLIR transformation passes. Transform IR is available through the dialect and CAPI."
  alias Beaver.{MLIR.CAPI, Native}
  use Beaver.ComposerGenerator, prefix: "mlirCreateTransforms"
  use Beaver.ComposerGenerator, prefix: "mlirCreateLinalg"
  use Beaver.ComposerGenerator, prefix: "mlirCreateGPU"

  @doc "Whether the linked LLVM supports packed tile-size and interchange parameters."
  def packed_params_supported?, do: CAPI.beaverTransformPackedParamsSupported() |> Native.to_term()
end
