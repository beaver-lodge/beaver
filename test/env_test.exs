defmodule EnvTest do
  use Beaver.Case, async: true

  test "MLIR.__BLOCK__", %{ctx: ctx} do
    function_module ctx, some_func(function_type: Type.function([], [Type.i(32)])) do
      %MLIR.Region{} = Beaver.Env.region()
      block(do: %MLIR.Block{} = Beaver.Env.block())
    end
  end

  test "location", %{ctx: ctx} do
    mlir ctx: ctx do
      loc = %MLIR.Location{} = MLIR.Location.from_env(__ENV__, ctx: Beaver.Env.context())
      assert loc |> MLIR.to_string() =~ ~r{env_test.exs:\d+:0}
    end
  end
end
