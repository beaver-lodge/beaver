defmodule EnvTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Func
  require Func

  test "MLIR.__BLOCK__", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            %MLIR.Region{} = Beaver.Env.region()

            block bb_entry() do
              %MLIR.Block{} = Beaver.Env.block()
            end
          end
        end
      end
    end
  end
end
