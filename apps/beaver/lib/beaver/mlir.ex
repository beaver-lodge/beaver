defmodule Beaver.MLIR do
  require Logger
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  alias Beaver.MLIR.CAPI.{
    MlirModule,
    MlirOperation,
    MlirAttribute,
    MlirBlock,
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

  def is_null(%MlirAttribute{} = v) do
    CAPI.beaverAttributeIsNull(v) |> CAPI.to_term()
  end

  def is_null(%MlirOperation{} = v) do
    CAPI.beaverOperationIsNull(v) |> CAPI.to_term()
  end

  def is_null(%MlirBlock{} = v) do
    CAPI.beaverBlockIsNull(v) |> CAPI.to_term()
  end

  def to_string(%MlirAttribute{ref: ref}) do
    CAPI.beaver_attribute_to_charlist(ref) |> CAPI.check!() |> List.to_string()
  end

  def to_string(%MlirOperation{ref: ref}) do
    CAPI.beaver_operation_to_charlist(ref) |> CAPI.check!() |> List.to_string()
  end

  def to_string(%MlirModule{} = module) do
    module |> __MODULE__.Operation.from_module() |> __MODULE__.to_string()
  end

  def to_string(%MlirType{ref: ref}) do
    CAPI.beaver_type_to_charlist(ref) |> CAPI.check!() |> List.to_string()
  end
end
