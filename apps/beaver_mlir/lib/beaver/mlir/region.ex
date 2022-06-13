defmodule Beaver.MLIR.Region do
  def create_blocks(region, f) when is_function(f, 0) do
    outer_region = Beaver.MLIR.Managed.Region.get()
    Beaver.MLIR.Managed.Region.set(region)
    blocks = f.()
    Beaver.MLIR.Managed.Region.set(outer_region)

    if not is_list(blocks), do: raise("Expected a list of blocks")

    for b <- blocks do
      Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, b)
    end

    Beaver.MLIR.Managed.Terminator.resolve()
    region
  end
end
