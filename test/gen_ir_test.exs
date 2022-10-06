defmodule CFTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  import ExUnit.CaptureIO

  test "generate mlir with function calls", context do
    ir =
      mlir ctx: context[:ctx] do
        module do
          Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
            region do
              block bb_entry() do
                v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                cond0 = Arith.constant(true) >>> Type.i(1)
                CF.cond_br(cond0, MLIR.__BLOCK__(bb1), {MLIR.__BLOCK__(bb2), [v0]}) >>> []
              end

              block bb1() do
                v1 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                _add = Arith.addi(v0, v0) >>> Type.i(32)
                CF.br({MLIR.__BLOCK__(bb2), [v1]}) >>> []
              end

              block bb2(arg >>> Type.i(32)) do
                v2 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                add = Arith.addi(arg, v2) >>> Type.i(32)
                Func.return(add) >>> []
              end
            end
          end
          |> MLIR.Operation.verify!(dump_if_fail: true)

          Func.func some_func2(function_type: Type.function([], [Type.i(32)])) do
            region do
              block bb_entry() do
                v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                _add = Arith.addi(v0, v0) >>> Type.i(32)
                CF.br({MLIR.__BLOCK__(bb1), [v0]}) >>> []
              end

              block bb1(arg >>> Type.i(32)) do
                v2 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                add = Arith.addi(arg, v2) >>> Type.i(32)
                _sub = Arith.subi(arg, v2) >>> Type.i(32)
                _mul = Arith.muli(arg, v2) >>> Type.i(32)
                _div = Arith.divsi(arg, v2) >>> Type.i(32)
                Func.return(add) >>> []
              end
            end
          end
        end
      end
      |> MLIR.Operation.verify!()

    captured =
      capture_io(fn ->
        IO.inspect(ir)
      end)

    assert captured =~ ~r"module"
    assert captured =~ ~r"// pred.+bb0"
    assert captured =~ ~r"// 2 preds.+bb0.+bb1"
  end
end