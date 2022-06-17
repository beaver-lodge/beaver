defmodule MemRefDescriptorTest do
  use ExUnit.Case

  test "test memref descriptor" do
    t =
      Beaver.MLIR.ExecutionEngine.MemRefDescriptor.create(
        [1, 2, 3, 4],
        [2, 2],
        [0, 0]
      )

    assert t
           |> Exotic.Value.fetch(Beaver.MLIR.ExecutionEngine.MemRefDescriptor, :offset)
           |> Exotic.Value.extract() == 0

    assert t
           |> Exotic.Value.fetch(Beaver.MLIR.ExecutionEngine.MemRefDescriptor, :shape)
           |> Exotic.Value.as_binary() == <<2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0>>

    assert t
           |> Exotic.Value.fetch(Beaver.MLIR.ExecutionEngine.MemRefDescriptor, :strides)
           |> Exotic.Value.as_binary() == <<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>>
  end
end
