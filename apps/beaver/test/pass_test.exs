defmodule PassTest do
  use ExUnit.Case
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.Func

  test "exception in run/1" do
    defmodule PassRaisingException do
      use Beaver.MLIR.Pass, on: Func.Func

      def run(func_) do
        raise "exception in pass run"
      end
    end

    ir =
      mlir do
        module do
          Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
            region do
              block bb_entry() do
                v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
                Func.return(v0) >>> []
              end
            end
          end
          |> MLIR.Operation.verify!(dump_if_fail: true)
        end
      end
      |> MLIR.Operation.verify!()

    assert_raise RuntimeError, ~r"Unexpected failure running pass pipeline", fn ->
      ir
      |> MLIR.Pass.Composer.nested(Func.Func, [
        PassRaisingException.create()
      ])
      |> MLIR.Pass.Composer.run!()
    end
  end
end
