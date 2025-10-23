defmodule Beaver.MLIR.InsertionPoint do
  @moduledoc """
  Functions to work with insertion point in Beaver DSL environment.
  """
  alias Beaver.MLIR

  @type t() :: MLIR.Block.t() | MLIR.PatternRewriter.t() | MLIR.RewriterBase.t()

  @doc """
  Convert insertion point to block if possible.
  """
  @spec to_block(t()) :: MLIR.Block.t() | nil
  def to_block(%MLIR.Block{} = blk), do: blk

  def to_block(%MLIR.PatternRewriter{} = rewriter) do
    MLIR.PatternRewriter.as_base(rewriter) |> to_block
  end

  def to_block(%MLIR.RewriterBase{} = rewriter_base) do
    MLIR.RewriterBase.block(rewriter_base)
  end

  def to_block(_), do: nil

  @doc """
  Insert operation at the given insertion point.
  """
  def insert_operation(%MLIR.Block{} = block, %MLIR.Operation{} = operation) do
    MLIR.Block.append(block, operation)
  end

  def insert_operation(%MLIR.RewriterBase{} = base, %MLIR.Operation{} = operation) do
    MLIR.RewriterBase.insert(base, operation)
  end

  def insert_operation(%MLIR.PatternRewriter{} = rewriter, %MLIR.Operation{} = operation) do
    MLIR.PatternRewriter.as_base(rewriter) |> MLIR.RewriterBase.insert(operation)
  end

  def insert_operation({:not_found, _}, _), do: nil

  def insert_operation(nil, _), do: nil
end
