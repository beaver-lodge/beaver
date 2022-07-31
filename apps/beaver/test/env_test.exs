defmodule EnvTest do
  use ExUnit.Case
  use Beaver
  alias Beaver.MLIR

  test "MLIR.__BLOCK__" do
    mlir do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            %MLIR.CAPI.MlirRegion{} = MLIR.__REGION__()

            block bb_entry() do
              %MLIR.CAPI.MlirBlock{} = MLIR.__BLOCK__()
            end
          end
        end
      end
    end
  end
end
