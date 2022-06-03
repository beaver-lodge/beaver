defmodule Beaver.MLIR.Transforms do
  @moduledoc """
  Transformations MLIR provides by default.
  """
  use Beaver.MLIR.Pass.Composer.Generator, prefix: "mlirCreateTransforms"
end
