for m <-
      [
        OpaqueArray,
        Bool,
        CInt,
        CUInt,
        F32,
        F64,
        I16,
        I32,
        I64,
        I8,
        ISize,
        U16,
        U32,
        U64,
        U8,
        USize
      ] do
  full = Module.concat(Beaver.Native, m)

  defmodule full do
    use Fizz.ResourceKind,
      forward_module: Beaver.Native
  end
end
