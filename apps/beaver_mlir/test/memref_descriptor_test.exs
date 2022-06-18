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
           |> Exotic.Value.fetch(
             Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
             :offset
           )
           |> Exotic.Value.extract() == 0

    assert t
           |> Exotic.Value.fetch(
             Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
             :shape
           )
           |> Exotic.Value.as_binary() == <<2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0>>

    assert t
           |> Exotic.Value.fetch(
             Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
             :strides
           )
           |> Exotic.Value.as_binary() == <<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>>

    assert t
           |> Exotic.Value.fetch(
             Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
             :aligned
           )
           |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 4, 8)) ==
             <<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0>>

    assert t
           |> Exotic.Value.fetch(
             Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(2),
             :allocated
           )
           |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 4, 8)) ==
             <<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0>>
  end
end
