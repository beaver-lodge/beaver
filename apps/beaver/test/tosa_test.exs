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
        Func.func test_multibroadcast([
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
    |> MLIR.Pass.Composer.nested("func.func", [tosa_to_linalg()])
    |> tosa_to_tensor()
    |> convert_func_to_llvm
    |> convert_arith_to_llvm
    |> MLIR.Pass.Composer.run!()
    |> MLIR.Operation.dump!()
  end
end
