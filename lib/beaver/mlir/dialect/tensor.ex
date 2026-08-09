defmodule Beaver.MLIR.Dialect.Tensor do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  alias Beaver.MLIR.{Attribute, Type}

  defmodule Slice do
    @moduledoc "Static/dynamic offsets, sizes, and strides for tensor slices."
    defstruct [:offsets, :sizes, :strides]
  end

  use Beaver.MLIR.Dialect,
    dialect: "tensor",
    ops: Beaver.MLIR.Dialect.Registry.ops("tensor")

  def reassociation(list) do
    for grouping <- list do
      grouping
      |> Enum.map(&Attribute.integer(Type.i64(), &1))
      |> Attribute.array()
    end
    |> Attribute.array()
  end

  def reassociation_for_reshape(src, target) do
    Beaver.MLIR.CAPI.beaverGetReassociationIndicesForReshape(src, target)
  end

  @doc "Build a validated mixed static/dynamic tensor slice specification."
  def slice(offsets, sizes, strides)
      when is_list(offsets) and is_list(sizes) and is_list(strides) do
    rank = length(offsets)

    unless length(sizes) == rank and length(strides) == rank do
      raise ArgumentError, "tensor slice offsets, sizes, and strides must have equal lengths"
    end

    %Slice{offsets: offsets, sizes: sizes, strides: strides}
  end

  @doc "Build `tensor.extract_slice` from a mixed static/dynamic slice spec."
  def extract_slice_(%Beaver.SSA{arguments: [source, %Slice{} = slice], ctx: ctx} = ssa) do
    arguments = slice_arguments(slice, ctx)

    Beaver.MLIR.Operation.eval_ssa(%Beaver.SSA{
      ssa
      | op: extract_slice(),
        arguments: [{:source, source} | arguments] ++ [operand_segment_sizes: :infer]
    })
  end

  @doc "Build `tensor.insert_slice` from a mixed static/dynamic slice spec."
  def insert_slice_(%Beaver.SSA{arguments: [source, dest, %Slice{} = slice], ctx: ctx} = ssa) do
    arguments = slice_arguments(slice, ctx)

    Beaver.MLIR.Operation.eval_ssa(%Beaver.SSA{
      ssa
      | op: insert_slice(),
        arguments:
          [{:source, source}, {:dest, dest} | arguments] ++ [operand_segment_sizes: :infer]
    })
  end

  defp slice_arguments(%Slice{} = slice, ctx) do
    {offsets, static_offsets} = split_slice_entries(slice.offsets, :offset)
    {sizes, static_sizes} = split_slice_entries(slice.sizes, :size)
    {strides, static_strides} = split_slice_entries(slice.strides, :stride)

    [
      {:offsets, offsets},
      {:sizes, sizes},
      {:strides, strides},
      {:static_offsets, Attribute.dense_array(static_offsets, Beaver.Native.I64, ctx: ctx)},
      {:static_sizes, Attribute.dense_array(static_sizes, Beaver.Native.I64, ctx: ctx)},
      {:static_strides, Attribute.dense_array(static_strides, Beaver.Native.I64, ctx: ctx)}
    ]
  end

  defp split_slice_entries(entries, kind) do
    Enum.reduce(entries, {[], []}, fn
      value, {dynamic, static} when is_integer(value) ->
        {dynamic, static ++ [value]}

      %Beaver.MLIR.Value{} = value, {dynamic, static} ->
        sentinel = Beaver.MLIR.ShapedType.to_dynamic_magic_number(:dynamic, kind)
        {dynamic ++ [value], static ++ [sentinel]}

      value, _acc ->
        raise ArgumentError,
              "tensor slice entries must be integers or index values, got: #{inspect(value)}"
    end)
  end
end
