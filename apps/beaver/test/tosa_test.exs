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
      |> MLIR.Pass.Composer.run!()
      |> convert_vector_to_llvm
      |> convert_memref_to_llvm
      |> convert_func_to_llvm
      |> reconcile_unrealized_casts
      |> MLIR.Pass.Composer.run!()

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

    return0 =
      MemRefDescriptor.create(
        [1.1, 2.2, 3.3] |> Enum.map(&Exotic.Value.get(:f32, &1)),
        [1, 3],
        [0, 0]
      )

    return =
      MemRefDescriptor.create(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6] |> Enum.map(&Exotic.Value.get(:f32, &1)),
        [2, 3],
        [0, 0]
      )

    for i <- 0..100 do
      _return =
        MLIR.ExecutionEngine.invoke!(
          jit,
          "test_multi_broadcast",
          [arg0, arg1] |> Enum.map(&Exotic.Value.get_ptr/1),
          Exotic.Value.get_ptr(return0)
        )
    end
  end
end
