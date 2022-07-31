defmodule Beaver.MLIR.Dialect.CF do
  alias Beaver.MLIR
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR.Dialect

  use Beaver.MLIR.Dialect,
    dialect: "cf",
    ops: Dialect.Registry.ops("cf"),
    skips: ~w{cond_br}

  defp sizes_of_block_args(arguments) do
    Enum.reduce(arguments, [], fn x, sizes ->
      case x do
        {%MLIR.CAPI.MlirBlock{}, block_args} when is_list(block_args) ->
          sizes ++ [length(block_args)]

        %MLIR.CAPI.MlirBlock{} ->
          sizes ++ [0]

        _ ->
          sizes
      end
    end)
  end

  defmodule CondBr do
    use Beaver.DSL.Op.Prototype, op_name: "cf.cond_br"
  end

  @doc """
  Create cf.cond_br op. Passing atom will lead to defer the creation of this terminator.
  """
  def cond_br(%Beaver.DSL.SSA{arguments: arguments, block: block}) do
    sizes = sizes_of_block_args(arguments)

    if length(sizes) not in [1, 2] do
      raise "cond_br requires 1 or 2 successors, instead got: #{length(sizes)}"
    end

    # always prepend 1 for the condition operand
    operand_segment_sizes = [1 | sizes] |> MLIR.ODS.operand_segment_sizes()

    MLIR.Operation.create(
      "cf.cond_br",
      arguments ++ [operand_segment_sizes: operand_segment_sizes],
      block
    )
  end
end
