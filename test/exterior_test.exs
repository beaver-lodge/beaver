defmodule ExteriorTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.{Func, Arith}
  alias Beaver.MLIR.Dialect.Elixir, as: Ex
  require Func

  test "generate ops in elixir dialect", %{ctx: ctx} do
    ir =
      mlir ctx: ctx do
        module do
          Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
            region do
              block do
                v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                _ = Ex.add(v0, v0) >>> Type.i(32)
                Func.return(v0) >>> []
              end
            end
          end
        end
        |> MLIR.Operation.verify!()
      end

    text = ir |> MLIR.to_string()
    assert text =~ "elixir.add"
  end
end
