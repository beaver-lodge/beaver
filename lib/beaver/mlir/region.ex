defmodule Beaver.MLIR.Region do
  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  def under(region, f) when is_function(f, 0) do
    f.()
    region
  end
end
