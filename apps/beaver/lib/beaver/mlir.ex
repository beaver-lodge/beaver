defmodule Beaver.MLIR do
  require Logger
  alias Beaver.MLIR.CAPI

  alias Beaver.MLIR.CAPI.{
    MlirModule,
    MlirOperation,
    MlirAttribute,
    MlirValue,
    MlirAffineExpr,
    MlirAffineMap,
    MlirIntegerSet,
    MlirType
  }

  def dump(%MlirModule{} = mlir) do
    CAPI.mlirModuleGetOperation(mlir)
    |> dump
  end

  def dump(%MlirOperation{} = mlir) do
    CAPI.mlirOperationDump(mlir)
    :ok
  end

  def dump(%MlirAttribute{} = mlir) do
    CAPI.mlirAttributeDump(mlir)
    :ok
  end

  def dump(%MlirValue{} = mlir) do
    CAPI.mlirValueDump(mlir)
    :ok
  end

  def dump(%MlirAffineExpr{} = mlir) do
    CAPI.mlirAffineExprDump(mlir)
    :ok
  end

  def dump(%MlirAffineMap{} = mlir) do
    CAPI.mlirAffineMapDump(mlir)
    :ok
  end

  def dump(%MlirIntegerSet{} = mlir) do
    CAPI.mlirIntegerSetDump(mlir)
    :ok
  end

  def dump(%MlirType{} = mlir) do
    CAPI.mlirTypeDump(mlir)
    :ok
  end

  def dump(_) do
    :error
  end

  def dump!(mlir) do
    with :ok <- dump(mlir) do
      mlir
    else
      :error ->
        error_msg = "can't dump #{inspect(mlir)}"
        raise error_msg
    end
  end
end
