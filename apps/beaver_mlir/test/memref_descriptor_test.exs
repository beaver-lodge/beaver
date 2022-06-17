defmodule MemRefDescriptorTest do
  use ExUnit.Case

  test "test memref descriptor" do
    Beaver.MLIR.ExecutionEngine.MemRefDescriptor.create(
      [1, 2, 3, 4],
      [2, 2],
      [0, 0]
    )
  end
end
