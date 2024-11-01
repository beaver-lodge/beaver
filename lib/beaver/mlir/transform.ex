defmodule Beaver.MLIR.Transform do
  @moduledoc """
  Transformations MLIR provides by default.
  """
  use Beaver.ComposerGenerator, prefix: "mlirCreateTransforms"
  use Beaver.ComposerGenerator, prefix: "mlirCreateLinalg"
  use Beaver.ComposerGenerator, prefix: "mlirCreateGPU"
end
