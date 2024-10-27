defmodule DebugOutputTest do
  use Beaver.Case, async: true
  use Beaver

  @moduletag :stderr
  @moduletag :smoke

  def ir(ctx) do
    ~m"""
    func.func @add(%arg0 : i32, %arg1 : i32) -> i32 {
      %res = arith.addi %arg0, %arg1 : i32
      return %res : i32
    }
    """.(ctx)
  end

  test "op stats", %{ctx: ctx} do
    ir(ctx)
    |> MLIR.Transforms.print_op_stats()
    |> MLIR.Pass.Composer.run!()
  end

  test "print ir", %{ctx: ctx} do
    ir(ctx)
    |> MLIR.Transforms.print_ir()
    |> MLIR.Pass.Composer.run!()
  end

  test "timing", %{ctx: ctx} do
    ir(ctx)
    |> MLIR.Transforms.canonicalize()
    |> MLIR.Pass.Composer.run!(timing: true)
  end

  test "pass manager enable print ir", %{ctx: ctx} do
    ir(ctx)
    |> MLIR.Transforms.canonicalize()
    |> MLIR.Pass.Composer.run!(print: true)
  end
end
