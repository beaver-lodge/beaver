defmodule TosaTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, TOSA}
  require Func

  test "generate and run tosa", context do
    import MLIR.{Transforms, Conversion}

    ir =
      mlir ctx: context[:ctx] do
        module do
          Func.func test_multi_broadcast(
                      function_type: ~a"(tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>"
                    ) do
            region do
              block entry(
                      arg0 >>> Type.ranked_tensor([1, 3], Type.f32()),
                      arg1 >>> Type.ranked_tensor([2, 1], Type.f32())
                    ) do
                v0 = TOSA.add(arg0, arg1) >>> Type.ranked_tensor([2, 3], Type.f32())

                v0 =
                  TOSA.mul(v0, arg1, {:shift, ~a{0 : i32}}) >>>
                    Type.ranked_tensor([2, 3], Type.f32())

                Func.return(v0) >>> []
              end
            end
          end
        end
      end
      |> MLIR.Operation.verify!()
      |> canonicalize
      |> cse
      |> tosa_to_scf
      |> tosa_to_arith
      |> tosa_to_tensor()
      |> convert_tensor_to_linalg()
      |> MLIR.Pass.Composer.nested("func.func", [
        tosa_to_linalg(),
        linalg_fuse_elementwise_ops(),
        linalg_bufferize(),
        convert_linalg_to_loops(),
        lower_affine(),
        convert_math_to_llvm(),
        convert_scf_to_cf(),
        "arith-expand",
        "memref-expand"
      ])
      |> MLIR.Pass.Composer.nested("func.func", fn pm ->
        MLIR.Pass.pipeline!(pm, "tensor-bufferize")
      end)
      |> MLIR.Pass.Composer.pipeline("func-bufferize")
      |> MLIR.Pass.Composer.nested("func.func", fn pm ->
        MLIR.Pass.pipeline!(pm, "llvm-request-c-wrappers")
      end)
      |> convert_vector_to_llvm
      |> convert_memref_to_llvm
      |> convert_func_to_llvm
      |> reconcile_unrealized_casts
      |> MLIR.Pass.Composer.run!()

    jit = ir |> MLIR.ExecutionEngine.create!()

    arg0 =
      Beaver.Native.Memory.new(
        [1.1, 2.2, 3.3],
        type: Beaver.Native.F32,
        sizes: [1, 3],
        strides: [0, 0]
      )

    arg1 =
      Beaver.Native.Memory.new(
        [1.1, 2.2],
        type: Beaver.Native.F32,
        sizes: [2, 1],
        strides: [0, 0]
      )

    <<
      a0::little-float-32,
      a1::little-float-32
    >> =
      arg1
      |> Beaver.Native.Memory.aligned()
      |> Beaver.Native.OpaquePtr.to_binary(Integer.floor_div(32 * 2, 8))

    assert [a0, a1] == [1.100000023841858, 2.200000047683716]

    return =
      Beaver.Native.Memory.new(
        [1.1, 2.2],
        type: Beaver.Native.F32,
        sizes: [2, 3],
        strides: [0, 0]
      )

    for _i <- 0..100 do
      # if return is a struct, it becomes first arg
      MLIR.ExecutionEngine.invoke!(
        jit,
        "test_multi_broadcast",
        Enum.map([return, arg0, arg1], &Beaver.Native.Memory.descriptor_ptr/1)
      )

      arg0
      |> Beaver.Native.Memory.aligned()
      |> Beaver.Native.OpaquePtr.to_binary(Integer.floor_div(32 * 3, 8))

      <<
        a0::little-float-32,
        a1::little-float-32
      >> =
        arg1
        |> Beaver.Native.Memory.aligned()
        |> Beaver.Native.OpaquePtr.to_binary(Integer.floor_div(32 * 2, 8))

      assert [a0, a1] == [1.100000023841858, 2.200000047683716]

      <<
        x0::little-float-32,
        x1::little-float-32,
        x2::little-float-32,
        x3::little-float-32,
        x4::little-float-32,
        x5::little-float-32
      >> =
        return
        # must use aligned ptr if it is allocated by LLVM
        |> Beaver.Native.Memory.aligned()
        |> Beaver.Native.OpaquePtr.to_binary(Integer.floor_div(32 * 6, 8))

      assert [x0, x1, x2, x3, x4, x5] == [
               2.4200000762939453,
               3.630000352859497,
               4.840000152587891,
               7.260000705718994,
               9.680000305175781,
               12.100000381469727
             ]

      #  TODO: deallocate with the memref descriptor returned
    end

    return.descriptor |> Beaver.Native.Memory.Descriptor.dump()
    assert return.descriptor |> Beaver.Native.Memory.Descriptor.offset() == 0
    Beaver.Native.Memory.descriptor_ptr(return) |> Beaver.Native.dump()
  end
end
