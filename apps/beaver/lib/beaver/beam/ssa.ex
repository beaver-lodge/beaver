defmodule Beaver.BEAM.SSA do
  @moduledoc """
  this module contains functions compiling BEAM SSA to MLIR.
  """
  def from_file!(path) do
    {:ok, tokens, _EndLocation} =
      File.read!(path)
      |> String.to_charlist()
      |> :erl_scan.string()

    {:ok, [form]} =
      tokens
      |> :erl_parse.parse_exprs()

    {:value, ssa, []} = :erl_eval.expr(form, [])
    ssa
  end
end
