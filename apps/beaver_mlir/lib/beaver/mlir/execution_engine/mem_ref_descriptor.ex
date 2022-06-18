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

  @doc """
  create descriptor without allocating elements, but with shape and strides. It is usually used as returned value.
  """
  def create(shape, strides) do
    create([], shape, strides)
  end

  @doc """
  create descriptor and allocate elements.
  """
  def create(elements, shape, strides)
      when is_list(elements) and is_list(shape) do
    rank = length(shape)

    if length(shape) != length(strides) do
      raise "elements and shape must have the same length"
    end

    arr_ptr =
      if elements == [] do
        Exotic.Value.Ptr.null()
      else
        elements
        |> Exotic.Value.Array.get()
        |> Exotic.Value.as_binary()
        |> IO.inspect(label: "as_binary")

        elements |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr()
      end

    aligned = allocated = arr_ptr

    offset = 0

    shape = shape |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    strides = strides |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    Exotic.Value.Struct.get(
      __MODULE__.struct_fields(rank),
      [allocated, aligned, offset, shape, strides]
    )
  end
end
