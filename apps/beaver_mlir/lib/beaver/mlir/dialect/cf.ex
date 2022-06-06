defmodule Beaver.MLIR.Dialect.CF do
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  defp extract_args(dest) when is_atom(dest) do
    {dest, []}
  end

  defp extract_args({dest, args}) when is_atom(dest) and is_list(args) do
    {dest, args}
  end

  @doc """
  Create cf.br op. It is a terminator, so this function doesn't returns the results
  """
  def br(dest) do
    {dest, args} = extract_args(dest)
    length_args = length(args)

    MLIR.Operation.create(
      "cf.br",
      [successor: dest] ++ args
    )

    nil
  end

  @doc """
  Create cf.cond_br op. It is a terminator, so this function doesn't returns the results
  """
  def cond_br(condition, true_dest, false_dest) do
    {true_dest, true_args} = extract_args(true_dest)
    {false_dest, false_args} = extract_args(false_dest)

    length_true_args = length(true_args)
    length_false_args = length(false_args)

    operand_segment_sizes =
      ~a{dense<[1, #{length_true_args}, #{length_false_args}]> : vector<3xi32>}

    MLIR.Operation.create(
      "cf.cond_br",
      [
        condition,
        successor: true_dest,
        operand_segment_sizes: operand_segment_sizes,
        successor: false_dest
      ] ++ true_args ++ false_args
    )

    nil
  end
end
