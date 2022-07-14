defmodule EnvTest do
  use ExUnit.Case
  use Beaver
  alias Beaver.MLIR

  test "mlir__BLOCK__" do
    mlir do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            %MLIR.CAPI.MlirRegion{} = Beaver.Env.mlir__REGION__()

            block bb_entry() do
              %MLIR.CAPI.MlirBlock{} = Beaver.Env.mlir__BLOCK__()
            end
          end
        end
      end
    end
  end
end
