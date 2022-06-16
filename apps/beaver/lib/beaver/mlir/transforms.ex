defmodule Beaver.MLIR.Transforms do
  @moduledoc """
  Transformations MLIR provides by default.
  """
  use Beaver.MLIR.Pass.Composer.Generator, prefix: "mlirCreateTransforms"
  use Beaver.MLIR.Pass.Composer.Generator, prefix: "mlirCreateLinalg"
end
