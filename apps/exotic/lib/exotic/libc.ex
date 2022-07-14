defmodule Exotic.LibC do
  use Exotic.Library, path: ["libm.so.6", "libm.so", "libSystem.B.dylib"]
  def puts(:ptr), do: :void
  def sin(:f64), do: :f64
  def cos(:f64), do: :f64
  @native [puts: 1, sin: 1, cos: 1]
end
