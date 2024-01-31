defmodule ASTIRTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR

  test "expander" do
    import Beaver.MLIR.AST

    defm do
      # single-block region
      var = MLIR."dialect.op1"()[attribute_name: 42 :: i32] :: {i1, i16, i32, i64}

      MLIR."dialect.op2"()["other attribute": 42 :: i64] ::
        i64 do
          # unnamed block
          _ ->
            MLIR."dialect.innerop2"() :: {}
            MLIR."dialect.innerop3"(var._0, var._2, var._3) :: {}
            bb1(var._0)

          bb1(arg :: i32) ->
            MLIR."dialect.innerop4"(arg)[attribute_name2: 42 :: i32] :: {}
            MLIR."dialect.innerop4-1"() :: {}

          bb2(arg2 :: i64) ->
            MLIR."dialect.innerop5"() :: {}
            MLIR."dialect.innerop5-1"(arg2) :: {}
        end
    end
    |> MLIR.dump!()
  end
end
