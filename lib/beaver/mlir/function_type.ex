defmodule Beaver.MLIR.FunctionType do
  @moduledoc """
  This module provides utilities for MLIR function type.
  """
  alias Beaver.MLIR
  import MLIR.CAPI

  defdelegate input(type, pos), to: MLIR.CAPI, as: :mlirFunctionTypeGetInput

  def num_inputs(type) do
    mlirFunctionTypeGetNumInputs(type) |> Beaver.Native.to_term()
  end

  defdelegate result(type, pos), to: MLIR.CAPI, as: :mlirFunctionTypeGetResult

  def num_results(type) do
    mlirFunctionTypeGetNumResults(type) |> Beaver.Native.to_term()
  end
end
