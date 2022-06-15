defmodule TosaTest do
  use ExUnit.Case
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "generate and run tosa" do
    require Beaver
    require Beaver.MLIR.Dialect.Func
    alias Beaver.MLIR
    alias Beaver.MLIR.Dialect.{Builtin, Func, Arith, CF, TOSA}
    import Builtin, only: :macros
    import MLIR, only: :macros
    import MLIR.Sigils

    Beaver.mlir do
      Builtin.module do
        Func.func test_multibroadcast(
                    function_type: ~a"(tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>"
                  ) do
          region do
            block entry(arg0 :: ~t{tensor<1x3xf32>}, arg1 :: ~t{tensor<2x1xf32>}) do
              v0 = TOSA.add(arg0, arg1) :: ~t{tensor<2x3xf32>}
              Func.return(v0)
            end
          end
        end
      end
    end
    |> MLIR.Operation.dump!()
    |> MLIR.Operation.verify!()
  end
end
