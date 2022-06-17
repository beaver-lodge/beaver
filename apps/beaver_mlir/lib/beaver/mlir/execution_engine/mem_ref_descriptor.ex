defmodule Beaver.MLIR.ExecutionEngine.MemRefDescriptor do
  use Exotic.Type.Struct,
    fields: [
      allocated: :ptr,
      aligned: :ptr,
      offset: :i64,
      shape: :ptr,
      strides: :ptr
    ]

  def create(elements, shape, strides) when is_list(elements) and is_list(shape) do
    arr = elements |> Exotic.Value.Array.get()
    arr_ptr = arr |> Exotic.Value.get_ptr()

    allocated = arr_ptr
    aligned = arr_ptr

    offset = 0

    shape_arr = shape |> Exotic.Value.Array.get()
    shape = shape_arr |> Exotic.Value.get_ptr()

    strides_arr = strides |> Exotic.Value.Array.get()
    strides = strides_arr |> Exotic.Value.get_ptr()

    Exotic.Value.Struct.get(
      __MODULE__,
      [allocated, aligned, offset, shape, strides]
    )
  end
end
