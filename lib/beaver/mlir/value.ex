defmodule Beaver.MLIR.Value do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

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
  Return the type of this value
  """
  defdelegate type(value), to: CAPI, as: :mlirValueGetType
end
