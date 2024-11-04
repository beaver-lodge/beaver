defmodule Beaver.MLIR.Value do
  @moduledoc """
  This module handles MLIR values, which represent SSA (Static Single Assignment) values in the IR.

  Values can be either block arguments or operation results. That's why this module provides
  functions to check if a value is an argument or a result (`argument?/1`, `result?/1`), or to get the owner of a result (`owner/1`).
  """
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind, forward_module: Beaver.Native

  def argument?(%__MODULE__{} = value) do
    CAPI.mlirValueIsABlockArgument(value) |> Beaver.Native.to_term()
  end

  @doc """
  Returns true if the value is a result of an operation.
  """
  def result?(%__MODULE__{} = value) do
    CAPI.mlirValueIsAOpResult(value) |> Beaver.Native.to_term()
  end

  @doc """
  Return the defining op of this value if this value is a result
  """
  def owner(%__MODULE__{} = value) do
    if result?(value) do
      {:ok, CAPI.mlirOpResultGetOwner(value)}
    else
      {:error, "not a result"}
    end
  end

  @doc """
  Return the defining op of this value. Raises if this value is not a result
  """
  def owner!(value) do
    case owner(value) do
      {:ok, op} ->
        op

      {:error, msg} ->
        raise ArgumentError, msg
    end
  end

  @doc """
  Return the type of this value
  """
  defdelegate type(value), to: CAPI, as: :mlirValueGetType
end
