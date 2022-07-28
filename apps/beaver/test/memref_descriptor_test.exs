defmodule MemRefDescriptorTest do
  use ExUnit.Case

  test "test memref descriptor" do
    t =
      Beaver.Native.Memory.new(
        [1, 2, 3, 4],
        sizes: [2, 2],
        type: Beaver.Native.I32
      )

    #  TODO: check offset
    #  TODO: check shape
    #  TODO: check strides

    assert t
           |> Beaver.Native.Memory.aligned()
           |> Beaver.Native.OpaquePtr.to_binary(Integer.floor_div(32 * 4, 8)) ==
             <<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0>>
  end
end
