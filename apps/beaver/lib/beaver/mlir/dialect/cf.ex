defmodule Beaver.MLIR.Dialect.CF do
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR.Dialect

  use Beaver.MLIR.Dialect,
    dialect: "cf",
    ops: Dialect.Registry.ops("cf"),
    skips: ~w{cond_br}

  defp extract_args(block = %Beaver.MLIR.CAPI.MlirBlock{}) do
    {:ok, {block, []}}
  end

  defp extract_args({dest = %Beaver.MLIR.CAPI.MlirBlock{}, args}) when is_list(args) do
    {:ok, {dest, args}}
  end

  defp extract_args(x) do
    {:other, x}
  end

  defp collect_arguments(arguments) do
    Enum.reduce(arguments, {[], []}, fn x, {arguments, seg_sizes} ->
      with {:ok, {dest, args}} <- extract_args(x) do
        arguments = arguments ++ args

        blocks = [
          case dest do
            dest when is_atom(dest) ->
              {:successor, dest}

            %MLIR.CAPI.MlirBlock{} ->
              dest
          end
        ]

        {arguments ++ blocks, seg_sizes ++ [length(args)]}
      else
        {:other, x} -> {arguments ++ [x], seg_sizes}
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
    {arguments, seg_sizes} = collect_arguments(arguments)

    if length(seg_sizes) not in [1, 2] do
      raise "cond_br requires 1 or 2 successors, but got seg_sizes: #{inspect(seg_sizes, pretty: true)}"
    end

    operand_segment_sizes =
      ~a{dense<[1, #{Enum.at(seg_sizes, 0, 0)}, #{Enum.at(seg_sizes, 1, 0)}]> : vector<3xi32>}

    MLIR.Operation.create(
      "cf.cond_br",
      arguments ++ [operand_segment_sizes: operand_segment_sizes],
      block
    )
  end
end
