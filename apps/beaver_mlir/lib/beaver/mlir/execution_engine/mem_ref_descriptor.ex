defmodule Beaver.MLIR.ExecutionEngine.MemRefDescriptor do
  @moduledoc """
  Get a memref descriptor's fields for Exotic Struct definition. Shape and strides will be omitted if rank is 0.
  """
  def struct_fields(0) do
    [
      allocated: :ptr,
      aligned: :ptr,
      offset: :i64
    ]
  end

  def struct_fields(rank) when is_integer(rank) and rank > 0 do
    sized_array = List.duplicate(:i64, rank)

    [
      allocated: :ptr,
      aligned: :ptr,
      offset: :i64,
      shape: {:struct, sized_array},
      strides: {:struct, sized_array}
    ]
  end

  # TODO: make this func public when it is possible to check ptr type
  defp create_from_ptr(ptr, shape, strides) do
    rank = length(shape)

    if length(shape) != length(strides) do
      raise "elements and shape must have the same length"
    end

    aligned = allocated = ptr

    offset = 0

    shape = shape |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    strides = strides |> Enum.map(&Exotic.Value.get(:i64, &1)) |> Exotic.Value.Array.get()

    Exotic.Value.Struct.get(
      __MODULE__.struct_fields(rank),
      [allocated, aligned, offset, shape, strides]
    )
  end

  @doc """
  create descriptor without allocating elements, but with shape and strides. It is usually used as returned value.
  """
  def create(shape, strides) do
    # TODO: extract a function
    create([], shape, strides)
  end

  @doc """
  create descriptor and allocate elements.
  """
  def create(elements, shape, strides)
      when is_list(elements) and is_list(shape) do
    arr_ptr =
      if elements == [] do
        Exotic.Value.Ptr.null()
      else
        elements |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr()
      end

    create_from_ptr(arr_ptr, shape, strides)
  end

  def create(binary, shape, strides) when is_binary(binary) do
    Exotic.Value.Struct.get(binary)
    |> Exotic.Value.get_ptr()
    |> create_from_ptr(shape, strides)
  end

  defp dense_stride(dims) when is_list(dims) and length(dims) > 0 do
    dims |> Enum.reduce(&*/2)
  end

  defp dense_strides([_], strides) when is_list(strides) do
    strides ++ [1]
  end

  defp dense_strides([_ | tail], strides) when is_list(strides) do
    dense_strides(tail, strides ++ [dense_stride(tail)])
  end

  def dense_strides([]) do
    []
  end

  def dense_strides(shape) when is_list(shape) do
    dense_strides(shape, [])
  end

  def read_as_binary(memref, len) when is_integer(len) do
    ptr =
      memref
      |> Exotic.Value.fetch(
        # rank here should not matter so set it to 1
        __MODULE__.struct_fields(1),
        :aligned
      )

    if Exotic.Value.extract(ptr) == Exotic.Value.extract(Exotic.Value.Ptr.null()) do
      raise "cannot read from memref of null pointer"
    end

    ptr |> Exotic.Value.Ptr.read_as_binary(len)
  end
end
