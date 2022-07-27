for m <-
      [
        OpaquePtr,
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
    alias Beaver.MLIR.CAPI

    use Fizz.ResourceKind,
      root_module: CAPI,
      forward_module: Beaver.Native

    if m == OpaquePtr do
      @doc """
      read the N bytes starting from the pointer and returns an Erlang binary
      """
      def read(%__MODULE__{ref: ref}, len) do
        CAPI.beaver_raw_read_opaque_ptr(ref, len)
      end
    end
  end
end
