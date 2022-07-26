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

    if m in [F32, F64, I32, I64] do
      def memref(
            allocated,
            aligned,
            offset,
            sizes,
            strides
          ) do
        apply(
          CAPI,
          Module.concat([__MODULE__, "memref_create"]) |> Beaver.Native.check!(),
          [allocated, aligned, offset, sizes, strides]
        )
      end

      def aligned(descriptor_ref) do
        ptr_ref =
          apply(
            CAPI,
            Module.concat([__MODULE__, "memref_aligned"]) |> Beaver.Native.check!(),
            [descriptor_ref]
          )

        %Beaver.Native.OpaquePtr{ref: ptr_ref}
      end
    end
  end
end
