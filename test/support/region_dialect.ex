defmodule TestRegion do
  @moduledoc "An example to showcase a region dialect in IRDL test."
  use Beaver.Slang, name: "test_region"

  defop any_region(i = {:single, Type.i32()}), do: [], regions: [:any]
  defop any_region2(i = {:single, Type.i32()}), do: [], regions: [:any, :any]
  defop sized_region(), regions: {:sized, 11}
  defop empty_args_region(), regions: {:region, args: []}

  defop typed_args_region(),
    regions: {:region, args: [Beaver.MLIR.Type.i32(), Beaver.MLIR.Type.i64()]}

  defop sized_typed_region(),
    regions: {:region, args: [Beaver.MLIR.Type.i32()], size: 2}
end
