defmodule ConversionSlang do
  @moduledoc false
  use Beaver.Slang, name: "conversion_test"

  defop source(), do: [Beaver.MLIR.Type.i32()]
  defop sink(value = Beaver.MLIR.Type.i32())
  defop keep()
end
