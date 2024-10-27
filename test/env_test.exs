defmodule EnvTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Func
  require Func

  test "MLIR.__BLOCK__", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            %MLIR.Region{} = Beaver.Env.region()

            block do
              %MLIR.Block{} = Beaver.Env.block()
            end
          end
        end
      end
    end
  end

  test "location", %{ctx: ctx} do
    mlir ctx: ctx do
      loc = %MLIR.Location{} = MLIR.Location.from_env(__ENV__, ctx: Beaver.Env.context())
      assert loc |> MLIR.to_string() =~ ~r{env_test.exs:\d+:0}
    end
  end
end
