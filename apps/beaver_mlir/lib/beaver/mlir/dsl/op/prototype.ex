defmodule Beaver.DSL.Op.Prototype do
  @moduledoc """
  Beaver.DSL.Prototype is a struct holding an Op's operands, results, attributes before creating it. It is similar to Beaver.DSL.SSA but there are applicable in different scenarios. Beaver.DSL.Prototype should be used where it is desired to get the created MLIR Op rather than the MLIR values of its results.
  Unlike SSA, operands/attributes/results in Prototype don't necessary contain real operands/attributes/results. They could be other types for abstraction or representation.
  For instance, when Prototype is used to compiling Elixir patterns to PDL, these fields contains MLIR Value.
  """

  defstruct operands: [], attributes: [], results: []

  @doc """
  Dispatch the op name and map to the callback `cb` if this is a module implement the behavior this module define.
  """
  def dispatch(module, fields, cb) when is_function(cb, 2) do
    dispatch(module, fields, :placeholder, fn n, p, :placeholder -> cb.(n, p) end)
  end

  def dispatch(module, fields, extra_arg, cb) when is_function(cb, 3) do
    if __MODULE__ in (module.module_info[:attributes][:behaviour] || []) do
      cb.(module.op_name(), struct!(Beaver.DSL.Op.Prototype, fields), extra_arg)
    else
      struct!(module, fields)
    end
  end

  @callback op_name() :: String.t()
end
