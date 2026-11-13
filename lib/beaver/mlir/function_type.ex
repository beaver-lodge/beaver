alias Beaver.MLIR.CAPI

defmodule Beaver.MLIR.FunctionType do
  @moduledoc "This module provides utilities for MLIR function type."
  import CAPI
  defdelegate input(type, pos), to: CAPI, as: :mlirFunctionTypeGetInput
  def num_inputs(type), do: mlirFunctionTypeGetNumInputs(type) |> Beaver.Native.to_term()
  defdelegate result(type, pos), to: CAPI, as: :mlirFunctionTypeGetResult
  def num_results(type), do: mlirFunctionTypeGetNumResults(type) |> Beaver.Native.to_term()
end
