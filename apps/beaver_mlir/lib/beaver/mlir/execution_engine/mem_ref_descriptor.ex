defmodule Beaver.MLIR.ExecutionEngine.MemRefDescriptor do
  def struct_fields(rank) when is_integer(rank) do
    sized_array = List.duplicate(:i64, rank)

    [
      allocated: :ptr,
      aligned: :ptr,
      offset: :i64,
      shape: {:struct, sized_array},
      strides: {:struct, sized_array}
    ]
  end

  def create(elements, shape, strides) when is_list(elements) and is_list(shape) do
    rank = length(shape)

    if length(shape) != length(strides) do
      raise "elements and shape must have the same length"
    end

    arr = elements |> Exotic.Value.Array.get()
    IO.inspect({shape, strides})
    arr |> Exotic.Value.as_binary() |> byte_size |> div(8) |> IO.inspect(label: "length")
    arr |> Exotic.Value.as_binary() |> IO.inspect()
    arr_ptr = arr |> Exotic.Value.get_ptr()

    allocated = arr_ptr
    aligned = arr_ptr

    offset = 0

    shape = shape |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    strides = strides |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    Exotic.Value.Struct.get(
      __MODULE__.struct_fields(rank),
      [allocated, aligned, offset, shape, strides]
    )
  end
end
