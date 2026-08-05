defmodule Beaver.MLIR.OpOperand do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @doc "Returns the operation that owns this operand use."
  @spec owner(t()) :: MLIR.Operation.t()
  defdelegate owner(op_operand), to: CAPI, as: :mlirOpOperandGetOwner

  @doc "Returns the value currently referenced by this operand use."
  @spec value(t()) :: MLIR.Value.t()
  defdelegate value(op_operand), to: CAPI, as: :mlirOpOperandGetValue

  @doc "Returns the operand position within the owner operation."
  @spec operand_number(t()) :: non_neg_integer()
  def operand_number(op_operand) do
    op_operand
    |> CAPI.mlirOpOperandGetOperandNumber()
    |> Beaver.Native.to_term()
  end
end
