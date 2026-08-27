defmodule OperationDestroyTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  test "iteratively destroys deeply nested operation trees", %{ctx: ctx} do
    depth = 128

    module =
      MLIR.Module.create!(
        String.duplicate("module { ", depth) <> String.duplicate(" }", depth),
        ctx: ctx
      )

    operation = MLIR.Operation.from_module(module)
    assert :ok = MLIR.CAPI.beaverOperationDestroyIterative_dirty_cpu(operation)
  end
end
