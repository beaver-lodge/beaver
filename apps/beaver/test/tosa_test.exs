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

    Beaver.mlir do
      module do
        Func.func test_multi_broadcast([
                    {:function_type, ~a"(tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>"},
                    {:"llvm.emit_c_interface", MLIR.Attribute.unit()}
                  ]) do
          region do
            block entry(arg0 :: ~t{tensor<1x3xf32>}, arg1 :: ~t{tensor<2x1xf32>}) do
              v0 = TOSA.add(arg0, arg1) :: ~t{tensor<2x3xf32>}
              Func.return(v0)
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
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run!()
    |> MLIR.Operation.dump!()
    |> MLIR.ExecutionEngine.create!()
  end
end
