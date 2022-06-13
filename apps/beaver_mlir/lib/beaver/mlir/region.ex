defmodule Beaver.MLIR.Region do
  def under(region, f) when is_function(f, 0) do
    outer_region = Beaver.MLIR.Managed.Region.get()
    Beaver.MLIR.Managed.Region.set(region)
    f.()
    Beaver.MLIR.Managed.Region.set(outer_region)

    Beaver.MLIR.Managed.Terminator.resolve()
    region
  end
end
