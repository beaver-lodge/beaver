defmodule Beaver.MLIR.ShapedType do
  @moduledoc """
  This module provides utilities for MLIR shaped types.
  """
  alias Beaver.MLIR
  import MLIR.CAPI

  def dynamic_stride_or_offset?(:dynamic), do: true

  def dynamic_stride_or_offset?(dim) do
    mlirShapedTypeIsDynamicStrideOrOffset(dim) |> Beaver.Native.to_term()
  end

  def static_stride_or_offset?(:dynamic), do: false

  def static_stride_or_offset?(dim) do
    mlirShapedTypeIsStaticStrideOrOffset(dim) |> Beaver.Native.to_term()
  end

  def dynamic_stride_or_offset() do
    mlirShapedTypeGetDynamicStrideOrOffset() |> Beaver.Native.to_term()
  end

  def dynamic_size() do
    mlirShapedTypeGetDynamicSize() |> Beaver.Native.to_term()
  end

  def dynamic_size?(:dynamic), do: true

  def dynamic_size?(size) do
    mlirShapedTypeIsDynamicSize(size) |> Beaver.Native.to_term()
  end

  def static_size?(:dynamic), do: false

  def static_size?(size) do
    mlirShapedTypeIsStaticSize(size) |> Beaver.Native.to_term()
  end

  def dynamic_dim?(type, dim) do
    mlirShapedTypeIsDynamicDim(type, dim) |> Beaver.Native.to_term()
  end

  def static_dim?(type, dim) do
    mlirShapedTypeIsStaticDim(type, dim) |> Beaver.Native.to_term()
  end

  def dim_size(type, dim) do
    mlirShapedTypeGetDimSize(type, dim)
    |> Beaver.Native.to_term()
    |> cast_dynamic_magic_number(:size)
  end

  def rank(type) do
    mlirShapedTypeGetRank(type) |> Beaver.Native.to_term()
  end

  def num_elements(type) do
    if not MLIR.Type.shaped?(type) do
      raise ArgumentError, "not a shaped type"
    end

    if not static?(type) do
      raise ArgumentError, "cannot get element count of dynamic shaped type"
    end

    beaverShapedTypeGetNumElements(type) |> Beaver.Native.to_term()
  end

  @doc """
  Returns whether the shape is fully static.
  """
  def static?(type) do
    mlirShapedTypeHasStaticShape(type) |> Beaver.Native.to_term()
  end

  def element_type(type) do
    if MLIR.Type.shaped?(type) do
      mlirShapedTypeGetElementType(type)
    else
      raise ArgumentError, "not a shaped type"
    end
  end

  @doc false
  def to_dynamic_magic_number(:dynamic, :size), do: dynamic_size()
  def to_dynamic_magic_number(:dynamic, :offset), do: dynamic_stride_or_offset()
  def to_dynamic_magic_number(:dynamic, :stride), do: dynamic_stride_or_offset()

  def to_dynamic_magic_number(size, _), do: size

  @doc false
  def cast_dynamic_magic_number(:dynamic, _), do: :dynamic

  def cast_dynamic_magic_number(stride_or_offset, modifier)
      when modifier in [:stride, :offset] do
    if MLIR.ShapedType.dynamic_stride_or_offset?(stride_or_offset) do
      :dynamic
    else
      stride_or_offset
    end
  end

  def cast_dynamic_magic_number(size, :size) do
    if MLIR.ShapedType.dynamic_size?(size) do
      :dynamic
    else
      size
    end
  end
end
