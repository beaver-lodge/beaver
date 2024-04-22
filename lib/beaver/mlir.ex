defmodule Beaver.MLIR do
  @moduledoc """
  This module provides functions to dump MLIR elements or print them to Elixir string.
  """
  require Logger
  alias Beaver.MLIR.CAPI

  alias Beaver.MLIR.{
    Value,
    Attribute,
    Type,
    Block,
    Location,
    Module,
    Operation,
    AffineMap,
    Dialect,
    AffineExpr,
    IntegerSet,
    OpPassManager,
    PassManager
  }

  defp dump_if_not_null(ir, dumper) do
    if is_null(ir) do
      {:error, "can't dump null"}
    else
      dumper.(ir)
    end
  end

  @type printable() ::
          Attribute.t()
          | Value.t()
          | Type.t()
          | Operation.t()
          | AffineMap.t()
          | Location.t()
          | OpPassManager.t()
          | PassManager.t()
          | Module.t()

  @type dump_opts :: [generic: boolean()]
  @spec dump(printable(), dump_opts()) :: :ok
  @spec dump!(printable(), dump_opts()) :: any()
  def dump(mlir, opts \\ [])

  def dump(%Module{} = mlir, opts) do
    CAPI.mlirModuleGetOperation(mlir)
    |> dump(opts)
  end

  def dump(%Operation{} = mlir, opts) do
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

  def dump(%AffineExpr{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirAffineExprDump/1)
  end

  def dump(%AffineMap{} = mlir, _opts) do
    dump_if_not_null(mlir, &CAPI.mlirAffineMapDump/1)
  end

  def dump(%IntegerSet{} = mlir, _opts) do
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

  @type nullable() ::
          Attribute.t()
          | Value.t()
          | Type.t()
          | Operation.t()
          | Module.t()
          | Block.t()
          | Dialect.t()

  @spec is_null(nullable()) :: boolean()

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

  def is_null(%Type{} = v) do
    CAPI.beaverTypeIsNull(v) |> Beaver.Native.to_term()
  end

  def is_null(%Dialect{} = v) do
    CAPI.beaverDialectIsNull(v) |> Beaver.Native.to_term()
  end

  @spec to_string(printable(), dump_opts()) :: :ok

  @doc """
  Print MLIR element as Elixir binary string. When printing an operation, it is recommended to use `generic: false` or `generic: true` to explicitly specify the format if your usage requires consistent output. If not specified, the default behavior is subject to change according to the MLIR version.
  """

  def to_string(mlir, opts \\ [])

  def to_string(%Attribute{ref: ref}, _opts) do
    CAPI.beaver_raw_to_string_attribute(ref) |> Beaver.Native.check!()
  end

  def to_string(%Value{ref: ref}, _opts) do
    CAPI.beaver_raw_to_string_value(ref) |> Beaver.Native.check!()
  end

  def to_string(%Operation{ref: ref}, opts) do
    generic = Keyword.get(opts, :generic)

    cond do
      opts[:bytecode] ->
        CAPI.beaver_raw_to_string_operation_bytecode(ref)

      generic == false ->
        CAPI.beaver_raw_to_string_operation_specialized(ref)

      generic == true ->
        CAPI.beaver_raw_to_string_operation_generic(ref)

      true ->
        CAPI.beaver_raw_to_string_operation(ref)
    end
    |> Beaver.Native.check!()
  end

  def to_string(%Module{} = module, opts) do
    module |> Operation.from_module() |> __MODULE__.to_string(opts)
  end

  def to_string(%Type{ref: ref}, _opts) do
    CAPI.beaver_raw_to_string_type(ref) |> Beaver.Native.check!()
  end

  def to_string(%AffineMap{ref: ref}, _opts) do
    CAPI.beaver_raw_to_string_affine_map(ref) |> Beaver.Native.check!()
  end

  def to_string(%Location{ref: ref}, _opts) do
    CAPI.beaver_raw_to_string_location(ref) |> Beaver.Native.check!()
  end

  def to_string(%OpPassManager{ref: ref}, _opts) do
    CAPI.beaver_raw_to_string_pm(ref) |> Beaver.Native.check!()
  end

  def to_string(%PassManager{} = pm, _opts) do
    pm |> CAPI.mlirPassManagerGetAsOpPassManager() |> __MODULE__.to_string()
  end
end

for m <- [
      Beaver.MLIR.Module,
      Beaver.MLIR.Attribute,
      Beaver.MLIR.Type,
      Beaver.MLIR.Value,
      Beaver.MLIR.Operation,
      Beaver.MLIR.AffineMap,
      Beaver.MLIR.Location,
      Beaver.MLIR.OpPassManager,
      Beaver.MLIR.PassManager
    ] do
  defimpl String.Chars, for: m do
    defdelegate to_string(mlir), to: Beaver.MLIR
  end
end
