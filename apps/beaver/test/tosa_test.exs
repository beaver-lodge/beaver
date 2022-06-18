defmodule TosaTest do
  use ExUnit.Case
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "generate and run tosa" do
    require Beaver
    require Beaver.MLIR.Dialect.Func
    alias Beaver.MLIR
    alias Beaver.MLIR.Dialect.{Builtin, Func, TOSA}
    import Builtin, only: :macros
    import MLIR, only: :macros
    import MLIR.Sigils
    import MLIR.{Transforms, Conversion}
    alias Beaver.MLIR.ExecutionEngine.MemRefDescriptor

    ir =
      Beaver.mlir do
        module do
          Func.func test_multi_broadcast([
                      {:function_type, ~a"(tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>"},
                      {:"llvm.emit_c_interface", MLIR.Attribute.unit()}
                    ]) do
            region do
              block entry(arg0 :: ~t{tensor<1x3xf32>}, arg1 :: ~t{tensor<2x1xf32>}) do
                v0 = TOSA.add(arg0, arg1) :: ~t{tensor<2x3xf32>}
                # v0 = TOSA.mul(arg0, arg1, {:shift, ~a{0 : i32}}) :: ~t{tensor<2x3xf32>}
                Func.return(v0)
                # Func.return(arg0)
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
      |> convert_vector_to_llvm
      |> convert_memref_to_llvm
      |> convert_func_to_llvm
      |> MLIR.Pass.Composer.run!()
      |> reconcile_unrealized_casts
      |> MLIR.Pass.Composer.run!()
      |> MLIR.Operation.dump!()

    jit = ir |> MLIR.ExecutionEngine.create!()

    arg0 =
      MemRefDescriptor.create(
        [1.1, 2.2, 3.3] |> Enum.map(&Exotic.Value.get(:f32, &1)),
        [1, 3],
        [0, 0]
      )

    arg1 =
      MemRefDescriptor.create(
        [1.1, 2.2] |> Enum.map(&Exotic.Value.get(:f32, &1)),
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

    return_ptr = Exotic.Value.get_ptr(return)

    for i <- 0..100 do
      MLIR.ExecutionEngine.invoke!(
        jit,
        "test_multi_broadcast",
        [return_ptr] ++ Enum.map([arg0, arg1], &Exotic.Value.get_ptr/1)
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
        |> Exotic.Value.fetch(
          Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
          :allocated
        )
        |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 6, 8))

      IO.inspect([x0, x1, x2, x3, x4, x5], label: i)
    end
  end
end
