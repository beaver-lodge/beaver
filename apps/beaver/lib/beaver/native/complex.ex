for m <-
      [
        F32
      ] do
  full = Module.concat(Beaver.Native.Complex, m)

  defmodule full do
    use Kinda.ResourceKind,
      forward_module: Beaver.Native
  end
end
