defmodule Beaver.MLIR.Value do
  alias Beaver.MLIR.CAPI

  def dump(value) do
    value |> CAPI.mlirValueDump()
  end

  def argument?(%CAPI.MlirValue{} = value) do
    CAPI.mlirValueIsABlockArgument(value) |> Exotic.Value.extract()
  end

  def result?(%CAPI.MlirValue{} = value) do
    CAPI.mlirValueIsAOpResult(value) |> Exotic.Value.extract()
  end

  def owner(value) do
    if result?(value) do
      {:ok, CAPI.mlirOpResultGetOwner(value)}
    else
      {:error, "not a result"}
    end
  end
end
