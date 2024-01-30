defmodule TestRegion do
  @moduledoc "An example to showcase the a region dialect in IRDL test."
  use Beaver.Slang, name: "test_region"
  defop any_region(i = {:single, Type.i32()}), do: [], regions: [:any]
  defop any_region2(i = {:single, Type.i32()}), do: [], regions: [:any, :any]
  defop sized_region(), regions: {:sized, 11}
end
