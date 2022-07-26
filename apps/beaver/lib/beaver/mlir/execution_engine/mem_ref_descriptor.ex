defmodule Beaver.MLIR.ExecutionEngine.MemRefDescriptor do
  alias Beaver.MLIR.CAPI

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
    shape = shape |> Beaver.Native.I64.array()
    strides = strides |> Beaver.Native.I64.array()

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
        nil
      else
        elements |> Beaver.Native.I64.array()
      end

    create_from_ptr(arr_ptr, shape, strides)
  end

  def create(binary, shape, strides) when is_binary(binary) do
    Exotic.Value.Struct.get(binary)
    |> Exotic.Value.get_ptr()
    |> create_from_ptr(shape, strides)
  end

  def read_as_binary(memref, len) when is_integer(len) do
    ptr =
      memref
      |> Exotic.Value.fetch(
        # rank here should not matter so set it to 1
        __MODULE__.struct_fields(1),
        :aligned
      )

    if Exotic.Value.extract(ptr) == Exotic.Value.extract(nil) do
      raise "cannot read from memref of null pointer"
    end

    ptr |> Exotic.Value.Ptr.read_as_binary(len)
  end
end
