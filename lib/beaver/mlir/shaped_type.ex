alias Beaver.Native

defmodule Beaver.MLIR.ShapedType do
  @moduledoc """
  This module provides utilities for MLIR shaped types.
  """
  alias Beaver.MLIR
  import MLIR.CAPI

  def dynamic_stride_or_offset?(:dynamic), do: true

  def dynamic_stride_or_offset?(dim), do: mlirShapedTypeIsDynamicStrideOrOffset(dim) |> Native.to_term()

  def static_stride_or_offset?(:dynamic), do: false

  def static_stride_or_offset?(dim), do: mlirShapedTypeIsStaticStrideOrOffset(dim) |> Native.to_term()

  def dynamic_stride_or_offset(), do: mlirShapedTypeGetDynamicStrideOrOffset() |> Native.to_term()

  def dynamic_size(), do: mlirShapedTypeGetDynamicSize() |> Native.to_term()

  def dynamic_size?(:dynamic), do: true

  def dynamic_size?(size), do: mlirShapedTypeIsDynamicSize(size) |> Native.to_term()

  def static_size?(:dynamic), do: false

  def static_size?(size), do: mlirShapedTypeIsStaticSize(size) |> Native.to_term()

  def dynamic_dim?(type, dim), do: mlirShapedTypeIsDynamicDim(type, dim) |> Native.to_term()

  def static_dim?(type, dim), do: mlirShapedTypeIsStaticDim(type, dim) |> Native.to_term()

  def dim_size(type, dim),
    do: mlirShapedTypeGetDimSize(type, dim) |> Native.to_term() |> cast_dynamic_magic_number(:size)

  def rank(type), do: mlirShapedTypeGetRank(type) |> Native.to_term()

  def num_elements(type) do
    if not MLIR.Type.shaped?(type), do: raise(ArgumentError, "not a shaped type")

    if not static?(type), do: raise(ArgumentError, "cannot get element count of dynamic shaped type")

    beaverShapedTypeGetNumElements(type) |> Native.to_term()
  end

  @doc """
  Returns whether the shape is fully static.
  """
  def static?(type), do: mlirShapedTypeHasStaticShape(type) |> Native.to_term()

  def element_type(type) do
    if MLIR.Type.shaped?(type), do: mlirShapedTypeGetElementType(type), else: raise(ArgumentError, "not a shaped type")
  end

  @doc false
  def to_dynamic_magic_number(:dynamic, :size), do: dynamic_size()
  def to_dynamic_magic_number(:dynamic, :offset), do: dynamic_stride_or_offset()
  def to_dynamic_magic_number(:dynamic, :stride), do: dynamic_stride_or_offset()

  def to_dynamic_magic_number(size, _), do: size

  @doc false
  def cast_dynamic_magic_number(:dynamic, _), do: :dynamic

  def cast_dynamic_magic_number(stride_or_offset, modifier) when modifier in [:stride, :offset],
    do: if(MLIR.ShapedType.dynamic_stride_or_offset?(stride_or_offset), do: :dynamic, else: stride_or_offset)

  def cast_dynamic_magic_number(size, :size),
    do: if(MLIR.ShapedType.dynamic_size?(size), do: :dynamic, else: size)
end
