defmodule Beaver.MLIR do
  @moduledoc """
  Provide macros to insert MLIR context and IR element of structure. These macros are designed to mimic the behavior and aesthetics of `__MODULE__`, `__CALLER__` in Elixir.
  Its distinguished form is to indicate this should not be expected to be a function or a macro works like a function.
  """
  require Logger
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  alias Beaver.MLIR.{
    Value,
    Attribute,
    Type,
    Block,
    Location,
    Module,
    Operation,
    AffineMap,
    Dialect
  }

  alias Beaver.MLIR.CAPI.{
    MlirAffineExpr,
    MlirIntegerSet,
    MlirOpPassManager,
    MlirPassManager
  }

  defp dump_if_not_null(ir, dumper) do
    if is_null(ir) do
      {:error, "can't dump null"}
    else
      dumper.(ir)
    end
  end

  @type dump_opts :: [generic: boolean()]
  @spec dump(any(), dump_opts()) :: :ok
  @spec dump!(any(), dump_opts()) :: any()
  def dump(mlir, opts \\ [])

  def dump(%__MODULE__.Module{} = mlir, opts) do
    CAPI.mlirModuleGetOperation(mlir)
    |> dump(opts)
  end

  def dump(%__MODULE__.Operation{} = mlir, opts) do
    if opts[:generic] do
      dump_if_not_null(mlir, &CAPI.beaverOperationDumpGeneric/1)
    else
      dump_if_not_null(mlir, &CAPI.mlirOperationDump/1)
    end
  end

  def dump(%Attribute{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirAttributeDump/1)
  end

  def dump(%Value{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirValueDump/1)
  end

  def dump(%MlirAffineExpr{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirAffineExprDump/1)
  end

  def dump(%AffineMap{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirAffineMapDump/1)
  end

  def dump(%MlirIntegerSet{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirIntegerSetDump/1)
  end

  def dump(%Type{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirTypeDump/1)
  end

  def dump(_, _) do
    {:error, "not a mlir element can be dumped"}
  end

  def dump!(mlir, opts \\ [])

  def dump!(mlir, opts) do
    case dump(mlir, opts) do
      :ok ->
        mlir

      {:error, msg} ->
        raise msg
    end
  end

  def is_null(%Attribute{} = v) do
    CAPI.beaverAttributeIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%Operation{} = v) do
    CAPI.beaverOperationIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%Module{} = m) do
    CAPI.beaverModuleIsNull(m) |> Beaver.Native.to_term()
  end

  def is_null(%Block{} = v) do
    CAPI.beaverBlockIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%Value{} = v) do
    CAPI.beaverValueIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%Dialect{} = v) do
    CAPI.beaverDialectIsNull(v) |> Beaver.Native.to_term()
  end

  def to_string(%Attribute{ref: ref}) do
    CAPI.beaver_raw_beaver_attribute_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%Value{ref: ref}) do
    CAPI.beaver_raw_beaver_value_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%Operation{ref: ref}) do
    CAPI.beaver_raw_beaver_operation_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%__MODULE__.Module{} = module) do
    module |> __MODULE__.Operation.from_module() |> __MODULE__.to_string()
  end

  def to_string(%Type{ref: ref}) do
    CAPI.beaver_raw_beaver_type_to_charlist(ref) |> Beaver.Native.check!() |> List.to_string()
  end

  def to_string(%AffineMap{ref: ref}) do
    CAPI.beaver_raw_beaver_affine_map_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%Location{ref: ref}) do
    CAPI.beaver_raw_beaver_location_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%MlirOpPassManager{ref: ref}) do
    CAPI.beaver_raw_beaver_pm_to_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end

  def to_string(%MlirPassManager{} = pm) do
    pm |> CAPI.mlirPassManagerGetAsOpPassManager() |> __MODULE__.to_string()
  end
end
