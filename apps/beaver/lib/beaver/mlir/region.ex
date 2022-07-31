defmodule Beaver.MLIR.Region do
  def under(region, f) when is_function(f, 0) do
    f.()

    Beaver.MLIR.Managed.Terminator.resolve()
    region
  end
end
