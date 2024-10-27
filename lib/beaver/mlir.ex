defmodule Beaver.MLIR do
  @moduledoc """
  This module provides common functions to work with MLIR entities.

  The functions in this module will leverage pattern matching to extract the entity type and call the corresponding CAPI function.
  """
  alias Beaver.MLIR.CAPI

  defp extract_entity_name(m) do
    ["Beaver", "MLIR", entity_name] = m |> Module.split()
    entity_name
  end

  alias Beaver.MLIR.{
    Module,
    Value,
    Attribute,
    Type,
    Block,
    Location,
    Operation,
    AffineMap,
    Dialect,
    OpPassManager,
    PassManager,
    Identifier,
    Diagnostic
  }

  @doc """
  Get the MLIR context of an MLIR entity.
  """
  def context(%m{} = entity) do
    entity_name = extract_entity_name(m)
    apply(CAPI, :"mlir#{entity_name}GetContext", [entity])
  end

  @doc """
  Get the MLIR location of an MLIR entity.
  """
  def location(%m{} = entity) do
    entity_name = extract_entity_name(m)
    apply(CAPI, :"mlir#{entity_name}GetLocation", [entity])
  end

  @doc """
  Compare two MLIR entities.
  """
  def equal?(a = %m{}, b = %m{}) do
    entity_name = extract_entity_name(m)
    apply(CAPI, :"mlir#{entity_name}Equal", [a, b]) |> Beaver.Native.to_term()
  end

  def equal?(a, b) when is_function(a, 1) do
    ctx = context(b)
    equal?(Beaver.Deferred.create(a, ctx), b)
  end

  def equal?(a, b) when is_function(b, 1) do
    ctx = context(a)
    equal?(a, Beaver.Deferred.create(b, ctx))
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
  def is_null(%m{} = entity) do
    entity_name = extract_entity_name(m)
    f = :"beaverIsNull#{entity_name}"
    function_exported?(CAPI, f, 1) && apply(CAPI, f, [entity]) |> Beaver.Native.to_term()
  end

  defp not_null_run(%m{} = entity, dumper) do
    entity_name = extract_entity_name(m)

    if is_null(entity) do
      {:error, "#{entity_name} is null"}
    else
      dumper.(entity)
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
          | Identifier.t()
          | Diagnostic.t()

  @type dump_opts :: [generic: boolean()]
  @spec dump(printable(), dump_opts()) :: :ok
  @spec dump!(printable(), dump_opts()) :: any()
  def dump(mlir, opts \\ [])

  def dump(%Beaver.MLIR.Module{} = mlir, opts) do
    CAPI.mlirModuleGetOperation(mlir)
    |> dump(opts)
  end

  def dump(%Operation{} = mlir, opts) do
    if opts[:generic] do
      not_null_run(mlir, &CAPI.beaverOperationDumpGeneric/1)
    else
      not_null_run(mlir, &CAPI.mlirOperationDump/1)
    end
  end

  def dump(%m{} = entity, _opts) do
    entity_name = extract_entity_name(m)
    not_null_run(entity, &apply(CAPI, :"mlir#{entity_name}Dump", [&1]))
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

  @spec to_string(printable(), dump_opts()) :: :ok

  @doc """
  Print MLIR element as Elixir binary string. When printing an operation, it is recommended to use `generic: false` or `generic: true` to explicitly specify the format if your usage requires consistent output. If not specified, the default behavior is subject to change according to the MLIR version.
  """

  def to_string(mlir, opts \\ [])

  def to_string(%Operation{ref: ref}, opts) do
    cond do
      opts[:bytecode] ->
        :Bytecode

      opts[:generic] == false ->
        :Specialized

      opts[:generic] == true ->
        :Generic

      true ->
        nil
    end
    |> then(&apply(CAPI, :"beaver_raw_to_string_Operation#{&1}", [ref]))
  end

  def to_string(%Beaver.MLIR.Module{} = module, opts) do
    module |> Operation.from_module() |> to_string(opts)
  end

  def to_string(%PassManager{} = pm, _opts) do
    pm |> CAPI.mlirPassManagerGetAsOpPassManager() |> __MODULE__.to_string()
  end

  def to_string(f, opts) when is_function(f) do
    Beaver.Deferred.create(f, Keyword.fetch!(opts, :ctx)) |> to_string(opts)
  end

  def to_string(%m{} = entity, _opts) do
    entity_name = extract_entity_name(m)
    not_null_run(entity, &apply(CAPI, :"beaver_raw_to_string_#{entity_name}", [&1.ref]))
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
      Beaver.MLIR.PassManager,
      Beaver.MLIR.Identifier,
      Beaver.MLIR.Diagnostic
    ] do
  defimpl String.Chars, for: m do
    defdelegate to_string(mlir), to: Beaver.MLIR
  end
end
