defmodule Beaver.MLIR.Dialect.CF do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect

  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """

  use Beaver.MLIR.Dialect,
    dialect: "cf",
    ops: Dialect.Registry.ops("cf")

  defp sizes_of_block_args(arguments) do
    Enum.reduce(arguments, [], fn x, sizes ->
      case x do
        {%MLIR.Block{}, block_args} when is_list(block_args) ->
          sizes ++ [length(block_args)]

        %MLIR.Block{} ->
          sizes ++ [0]

        _ ->
          sizes
      end
    end)
  end

  @doc """
  Create cf.cond_br op. Passing atom will lead to defer the creation of this terminator.
  """
  def cond_br(%Beaver.SSA{arguments: arguments, results: []} = ssa) do
    sizes = sizes_of_block_args(arguments)

    if length(sizes) not in [1, 2] do
      raise "cond_br requires 1 or 2 successors, instead got: #{length(sizes)}"
    end

    # always prepend 1 for the condition operand
    operand_segment_sizes = [1 | sizes] |> MLIR.ODS.operand_segment_sizes()

    MLIR.Operation.create(%Beaver.SSA{
      ssa
      | op: "cf.cond_br",
        arguments: arguments ++ [operand_segment_sizes: operand_segment_sizes]
    })
  end
end
