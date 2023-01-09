defmodule Beaver.MLIR.Region do
  use Kinda.ResourceKind,
    forward_module: Beaver.Native,
    fields: [safe_to_print: true]

  def under(region, f) when is_function(f, 0) do
    f.()
    region
  end
end
