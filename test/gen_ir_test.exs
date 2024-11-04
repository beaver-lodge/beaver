defmodule CFTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Func, Arith, CF}
  require Func

  test "generate mlir with function calls", %{ctx: ctx} do
    ir =
      mlir ctx: ctx do
        module do
          unquote(File.read!("test/readme_example.exs") |> Code.string_to_quoted!())

          Func.func some_func2(function_type: Type.function([], [Type.i(32)])) do
            region do
              block do
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
      |> MLIR.verify!()

    text = ir |> MLIR.to_string()

    assert text =~ ~r"module"
    assert text =~ ~r"// pred.+bb0"
    assert text =~ ~r"// 2 preds.+bb0.+bb1"
  end
end
