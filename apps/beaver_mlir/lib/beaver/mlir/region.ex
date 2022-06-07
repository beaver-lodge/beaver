defmodule Beaver.MLIR.Region do
  def create_blocks(region, f) when is_function(f, 0) do
    blocks = f.()
    if not is_list(blocks), do: raise("Expected a list of blocks")

    for b <- blocks do
      Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, b)
    end

    Beaver.MLIR.Managed.Terminator.resolve()
    region
  end
end
