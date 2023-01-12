defmodule EnvTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Func
  require Func

  test "MLIR.__BLOCK__", context do
    mlir ctx: context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            %MLIR.Region{} = MLIR.__REGION__()

            block bb_entry() do
              %MLIR.Block{} = MLIR.__BLOCK__()
            end
          end
        end
      end
    end
  end
end
