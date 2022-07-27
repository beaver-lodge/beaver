for m <-
      [
        F32
      ] do
  full = Module.concat(Beaver.Native.Complex, m)

  defmodule full do
    alias Beaver.MLIR.CAPI

    use Fizz.ResourceKind,
      root_module: CAPI,
      forward_module: Beaver.Native
  end
end
