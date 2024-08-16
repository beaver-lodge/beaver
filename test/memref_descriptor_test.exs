defmodule MemRefDescriptorTest do
  use ExUnit.Case, async: true
  alias Beaver.Native

  test "test memref descriptor" do
    sizes = [2, 2]

    t =
      Native.Memory.new(
        [1, 2, 3, 4],
        sizes: sizes,
        type: Native.I32
      )

    assert t.descriptor |> Native.Memory.Descriptor.offset() == 0
    assert t.descriptor |> Native.Memory.Descriptor.sizes() == sizes
    assert t.descriptor |> Native.Memory.Descriptor.strides() == [2, 1]

    assert t
           |> Native.Memory.aligned()
           |> Native.OpaquePtr.to_binary(Integer.floor_div(32 * 4, 8)) ==
             <<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0>>
  end
end
