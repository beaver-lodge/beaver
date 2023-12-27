defmodule CFTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Func, Arith, CF}
  require Func

  test "generate mlir with function calls", test_context do
    ir =
      mlir ctx: test_context[:ctx] do
        module do
          Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
            region do
              block bb_entry() do
                v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                cond0 = Arith.constant(true) >>> Type.i(1)
                CF.cond_br(cond0, Beaver.Env.block(bb1), {Beaver.Env.block(bb2), [v0]}) >>> []
              end

              block bb1() do
                v1 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                _add = Arith.addi(v0, v0) >>> Type.i(32)
                CF.br({Beaver.Env.block(bb2), [v1]}) >>> []
              end

              block bb2(arg >>> Type.i(32)) do
                v2 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                add = Arith.addi(arg, v2) >>> Type.i(32)
                Func.return(add) >>> []
              end
            end
          end
          |> MLIR.Operation.verify!(debug: true)

          Func.func some_func2(function_type: Type.function([], [Type.i(32)])) do
            region do
              block bb_entry() do
                v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                _add = Arith.addi(v0, v0) >>> Type.i(32)
                CF.br({Beaver.Env.block(bb1), [v0]}) >>> []
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

    text = ir |> MLIR.to_string()

    assert text =~ ~r"module"
    assert text =~ ~r"// pred.+bb0"
    assert text =~ ~r"// 2 preds.+bb0.+bb1"
  end

  test "expander" do
    File.read!("test/expander_base_example.exs")
    |> Code.string_to_quoted()
    |> Macro.postwalk(fn
      {:when, [], ast} -> dbg(ast)
      ast -> ast
    end)

    quote do
      # single-block region
      mast do
        MLIR."builtin.module" do
          var = MLIR."dialect.op1"()[attribute_name: 42 :: i32] :: {i1, i16, i32, i64}

          MLIR."dialect.op2" "other attribute": 42 :: i64 do
            # unnamed block
            _ ->
              MLIR."dialect.innerop2"() :: {}
              MLIR."dialect.innerop3"(var._0, var._2, var._3) :: {}
              bb1(var._0)

            bb1(arg :: i32) ->
              MLIR."dialect.innerop4"() :: {}

            bb2(arg :: i64) ->
              MLIR."dialect.innerop4"() :: {}
          end
        end
      end
    end
    |> dbg
  end
end
