defmodule DebugOutputTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR

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

  test "op stats", test_context do
    ir(test_context[:ctx])
    |> MLIR.Transforms.print_op_stats()
    |> MLIR.Pass.Composer.run!()
  end

  test "print ir", test_context do
    ir(test_context[:ctx])
    |> MLIR.Transforms.print_ir()
    |> MLIR.Pass.Composer.run!()
  end

  test "timing", test_context do
    ir(test_context[:ctx])
    |> MLIR.Transforms.canonicalize()
    |> MLIR.Pass.Composer.run!(timing: true)
  end
end
