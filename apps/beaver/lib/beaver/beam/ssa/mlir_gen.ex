defmodule Beaver.BEAM.SSA.MLIRGen do
  @moduledoc """
  Generate MLIR from different elements of BEAM SSA.
  """

  def module({:b_module, %{}, _module_name, _exports, [], functions}) do
    functions |> Enum.map(&function/1)
  end

  def function(
        {:b_function,
         %{
           func_info: {_module_name, _function_name, arity}
         }, args, _blocks, _count}
      ) do
    if length(args) != arity, do: raise("arity mismatch")
  end
end
