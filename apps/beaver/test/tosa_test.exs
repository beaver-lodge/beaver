defmodule TosaTest do
  use ExUnit.Case, async: true
  use Beaver

  test "generate and run tosa" do
    import MLIR.{Transforms, Conversion}
    alias Beaver.MLIR.ExecutionEngine.MemRefDescriptor

    ir =
      mlir do
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
      MemRefDescriptor.create(
        [1.1, 2.2, 3.3] |> Beaver.Native.F32.array(),
        [1, 3],
        [0, 0]
      )

    arg1 =
      MemRefDescriptor.create(
        [1.1, 2.2] |> Beaver.Native.F32.array(),
        [2, 1],
        [0, 0]
      )

    <<
      a0::little-float-32,
      a1::little-float-32
    >> =
      arg1
      |> Exotic.Value.fetch(
        Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
        :allocated
      )
      |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 2, 8))

    assert [a0, a1] == [1.100000023841858, 2.200000047683716]

    return =
      MemRefDescriptor.create(
        [2, 3],
        [0, 0]
      )

    for _i <- 0..100 do
      # if return is a struct, it becomes first arg
      MLIR.ExecutionEngine.invoke!(
        jit,
        "test_multi_broadcast",
        Enum.map([return, arg0, arg1], &Exotic.Value.get_ptr/1)
      )

      arg0
      |> Exotic.Value.fetch(
        Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
        :allocated
      )
      |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 3, 8))

      <<
        a0::little-float-32,
        a1::little-float-32
      >> =
        arg1
        |> Exotic.Value.fetch(
          Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
          :allocated
        )
        |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 2, 8))

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
        |> Exotic.Value.fetch(
          Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
          :aligned
        )
        |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 6, 8))

      assert [x0, x1, x2, x3, x4, x5] == [
               2.4200000762939453,
               3.630000352859497,
               4.840000152587891,
               7.260000705718994,
               9.680000305175781,
               12.100000381469727
             ]

      assert return
             |> Exotic.Value.fetch(
               Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
               :offset
             )
             |> Exotic.Value.extract() == 0
    end
  end
end
