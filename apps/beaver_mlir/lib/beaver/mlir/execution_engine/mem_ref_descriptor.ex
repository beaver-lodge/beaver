defmodule Beaver.MLIR.ExecutionEngine.MemRefDescriptor do
  use Exotic.Type.Struct,
    fields: [
      allocated: :ptr,
      aligned: :ptr,
      offset: :i64,
      shape: {:struct, [:i64, :i64]},
      strides: {:struct, [:i64, :i64]}
    ]

  def create(elements, shape, strides) when is_list(elements) and is_list(shape) do
    arr = elements |> Exotic.Value.Array.get()
    arr_ptr = arr |> Exotic.Value.get_ptr()

    allocated = arr_ptr
    aligned = arr_ptr

    offset = 0

    shape = shape |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    strides = strides |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    Exotic.Value.Struct.get(
      __MODULE__,
      [allocated, aligned, offset, shape, strides]
    )
  end
end
