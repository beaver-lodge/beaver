defmodule Beaver.MLIR.Conversion do
  @moduledoc """
  Conversions MLIR provides by default.
  """
  use Beaver.MLIR.Pass.Composer.Generator, prefix: "mlirCreateConversion"
end
