defmodule Beaver.MLIR.Region do
  def create_blocks(region, f) when is_function(f, 1) do
    f.(region)
  end
end
