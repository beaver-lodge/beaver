alias Beaver.MLIR.{Block, Operation, RewriterBase}

defmodule Beaver.MLIR.InsertionPoint do
  @moduledoc "Functions to work with insertion point in Beaver DSL environment."
  alias Beaver.MLIR
  @type t() :: Block.t() | MLIR.PatternRewriter.t() | RewriterBase.t()
  @doc "Convert insertion point to block if possible."
  @spec to_block(t()) :: Block.t() | nil
  def to_block(%Block{} = blk), do: blk
  def to_block(%MLIR.PatternRewriter{} = rewriter), do: MLIR.PatternRewriter.as_base(rewriter) |> to_block
  def to_block(%RewriterBase{} = rewriter_base), do: RewriterBase.block(rewriter_base)
  def to_block(_), do: nil
  @doc "Insert operation at the given insertion point."
  def insert_operation(%Block{} = block, %Operation{} = op), do: Block.append(block, op)
  def insert_operation(%RewriterBase{} = base, %Operation{} = op), do: RewriterBase.insert(base, op)

  def insert_operation(%MLIR.PatternRewriter{} = rewriter, %Operation{} = op),
    do: MLIR.PatternRewriter.as_base(rewriter) |> RewriterBase.insert(op)

  def insert_operation({:not_found, _}, _), do: nil
  def insert_operation(nil, _), do: nil
end
