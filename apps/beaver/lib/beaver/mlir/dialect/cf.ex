defmodule Beaver.MLIR.Dialect.CF do
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  defp extract_args(block = %Beaver.MLIR.CAPI.MlirBlock{}) do
    {:ok, {block, []}}
  end

  defp extract_args(dest) when is_atom(dest) do
    {:ok, {dest, []}}
  end

  defp extract_args({dest, args}) when is_atom(dest) and is_list(args) do
    {:ok, {dest, args}}
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
        arguments = arguments ++ [successor: dest] ++ args
        {arguments, seg_sizes ++ [length(args)]}
      else
        {:other, x} -> {arguments ++ [x], seg_sizes}
      end
    end)
  end

  @doc """
  Create cf.br op. It is a terminator, so this function doesn't returns the results
  """
  def br(%Beaver.DSL.SSA{arguments: arguments, block: block}) do
    {arguments, _} = collect_arguments(arguments)

    MLIR.Operation.create(
      "cf.br",
      arguments,
      block
    )

    nil
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

  def ops() do
    []
  end
end
