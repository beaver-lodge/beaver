defmodule CfTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  @moduletag :smoke

  test "cf with mutation", %{ctx: ctx} do
    import MutCompiler

    mlir =
      defnative get_lr(total_iters, factor, base_lr, step) do
        base_lr = base_lr * factor

        return(base_lr)
      end

    assert mlir =~ "%0 = arith.mulf %arg2, %arg1 : f32", mlir

    mlir =
      defnative get_lr_with_ctrl_flow(total_iters, factor, base_lr, step) do
        base_lr =
          if step < total_iters do
            base_lr * factor
          else
            base_lr
          end

        return(base_lr)
      end

    ir = ~m{#{mlir}}.(ctx)

    f = get_func(ir, "get_lr_with_ctrl_flow")

    assert f.(1000.0, 0.5, 0.002, 200.0) /
             f.(1000.0, 0.5, 0.002, 2000.0) == 0.5

    assert mlir =~ "%1 = arith.mulf %arg2, %arg1 : f32", mlir
    assert mlir =~ "return %2 : f32", mlir
  end
end
