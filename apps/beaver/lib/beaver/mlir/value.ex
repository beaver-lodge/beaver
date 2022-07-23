defmodule Beaver.MLIR.Value do
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  def dump(value) do
    value |> CAPI.mlirValueDump()
  end

  def argument?(%CAPI.MlirValue{} = value) do
    CAPI.mlirValueIsABlockArgument(value) |> CAPI.to_term()
  end

  @doc """
  Returns true if the value is a result of an operation.
  """
  def result?(%CAPI.MlirValue{} = value) do
    CAPI.mlirValueIsAOpResult(value) |> CAPI.to_term()
  end

  @doc """
  Return the defining op of this value if this value is a result
  """
  def owner(value) do
    if result?(value) do
      {:ok, CAPI.mlirOpResultGetOwner(value)}
    else
      {:error, "not a result"}
    end
  end
end
