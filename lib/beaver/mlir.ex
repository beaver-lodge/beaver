defmodule Beaver.MLIR do
  alias Beaver.MLIR.{Attribute, Type, Operation}
  alias Beaver.MLIR
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Core functionality for working with MLIR.

  This module serves as the primary interface for MLIR operations and entities in Beaver. It provides:

  - String conversion utilities for debugging and serialization
  - Null checking for safe operation handling
  - Context and location retrieval for MLIR entities

  ## Printing Operations

  When converting operations to strings, you can specify the output format:
  - `generic: true` - Uses MLIR's generic format
  - `generic: false` - Uses MLIR's specialized format
  - `bytecode: true` - Uses MLIR's bytecode format

  The default format may vary between MLIR versions, so explicitly specify the format
  when consistent output is required.

  ## Inspecting MLIR Entities
  Beaver doesn't implement `Inspect` protocol for MLIR entities because the output might be too verbose and can crash the BEAM if invalid entities are passed. Use `Beaver.MLIR.to_string/2` to convert entities to inspect it.

  ## Null Safety

  Many MLIR operations can return null values. Use `null?/1` to safely check entities
  before performing operations that require non-null values.

  ## Name spaces to include different kinds of CAPI delegates
  - `Beaver.MLIR.***`: APIs related to lifecycle, including creating and destroying MLIR entities.
  - `Beaver.MLIR`: APIs like `Beaver.MLIR.dump!/1` or `Beaver.MLIR.null?/1`. These are standard features generally expected in any MLIR tools.
  """

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
    ExecutionEngine
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
  def equal?(%m{} = a, %m{} = b) do
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
          | ExecutionEngine.t()

  @doc """
  Check if an MLIR entity is null.

  To prevent crashing the BEAM, it is encouraged to use this function to check if an entity is null before calling functions that require a non-null entity.
  """
  @spec null?(nullable()) :: boolean()
  def null?(%m{} = entity) do
    entity_name = extract_entity_name(m)
    f = :"beaverIsNull#{entity_name}"
    function_exported?(CAPI, f, 1) && apply(CAPI, f, [entity]) |> Beaver.Native.to_term()
  end

  defp not_null_run(%m{} = entity, dumper) do
    entity_name = extract_entity_name(m)

    if null?(entity) do
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

  @type dump_opts :: [generic: boolean()]
  @spec dump(printable(), dump_opts()) :: :ok
  @spec dump!(printable(), dump_opts()) :: any()
  @doc """
  Dump MLIR element to stdio.

  This will call the printer registered in C/C++. Note that the outputs wouldn't go through Erlang's IO system, so it's not possible to capture the output in Elixir. If you need to capture the output, use `to_string/1` instead.
  """
  def dump(mlir, opts \\ [])

  def dump(%Beaver.MLIR.Module{} = mlir, opts) do
    mlirModuleGetOperation(mlir) |> dump(opts)
  end

  def dump(%Operation{} = mlir, opts) do
    if opts[:generic] do
      not_null_run(mlir, &beaverOperationDumpGeneric/1)
    else
      not_null_run(mlir, &mlirOperationDump/1)
    end
  end

  def dump(%m{} = entity, _opts) do
    entity_name = extract_entity_name(m)
    not_null_run(entity, &apply(CAPI, :"mlir#{entity_name}Dump", [&1]))
  end

  @doc """
  Dump MLIR element to stdio and raise an error if it fails.
  """
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
  Print MLIR element or StringRef as Elixir binary string.

  When printing an operation, it is recommended to use `generic: false` or `generic: true` to explicitly specify the format if your usage requires consistent output. If not specified, the default behavior is subject to change according to the MLIR version.
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
    pm |> mlirPassManagerGetAsOpPassManager() |> __MODULE__.to_string()
  end

  def to_string(f, opts) when is_function(f) do
    Beaver.Deferred.create(f, Keyword.fetch!(opts, :ctx)) |> to_string(opts)
  end

  def to_string(%m{} = entity, _opts) do
    entity_name = extract_entity_name(m)
    not_null_run(entity, &apply(CAPI, :"beaver_raw_to_string_#{entity_name}", [&1.ref]))
  end

  @type verifiable() :: Operation.t() | Module.t()
  @spec verify!(verifiable()) :: :ok
  def verify!(op) do
    case verify(op) do
      {:ok, op} ->
        op

      :null ->
        raise "MLIR operation verification failed because the operation is null. Maybe it is parsed from an ill-formed text format? Please have a look at the diagnostic output above by MLIR C++"

      {:error, diagnostics} ->
        raise MLIR.Diagnostic.format(diagnostics, "MLIR operation verification failed")
    end
  end

  @spec verify(verifiable()) :: {:ok, verifiable()} | :null | {:error, list()}
  def verify(op) do
    if null?(op) do
      :null
    else
      ctx = MLIR.context(op)

      {is_success, diagnostics} =
        mlirOperationVerifyWithDiagnostics(ctx, MLIR.Operation.from_module(op))

      if Beaver.Native.to_term(is_success) do
        {:ok, op}
      else
        {:error, diagnostics}
      end
    end
  end

  @type applicable() :: Operation.t() | Module.t()
  @type apply_opt :: {:debug, boolean()}
  @apply_default_opts [debug: false]
  @spec apply!(applicable(), apply_opt) :: applicable()
  @doc """
  Apply patterns on a container (region, operation, module).
  It returns the container if it succeeds otherwise it raises.
  """
  def apply!(op, patterns, opts \\ @apply_default_opts) do
    case apply_(op, patterns, opts) do
      {:ok, module} ->
        module

      {:error, msg} ->
        raise msg
    end
  end

  @doc """
  Apply patterns on a container (operation, module).
  It is named `apply_` with a underscore to avoid name collision with `Kernel.apply/2`
  """
  @spec apply_(applicable(), apply_opt) :: {:ok, applicable()} | {:error, String.t()}

  def apply_(op, patterns, opts \\ @apply_default_opts)

  def apply_(op, %MLIR.FrozenRewritePatternSet{} = patterns, _opts) do
    MLIR.verify!(op)

    {result, diagnostics} =
      Beaver.MLIR.Rewrite.apply_patterns(op, patterns)

    if MLIR.LogicalResult.success?(result) do
      {:ok, op}
    else
      {:error, MLIR.Diagnostic.format(diagnostics, "failed to apply pattern set.")}
    end
  end

  def apply_(op, patterns, opts) do
    ctx = MLIR.Operation.from_module(op) |> MLIR.context()
    opts = Keyword.put_new(opts, :ctx, ctx)
    {set, pdl_mod} = MLIR.RewritePatternSet.with_pdl_patterns(patterns, opts)
    frozen_set = set |> MLIR.RewritePatternSet.freeze()

    apply_(op, frozen_set, opts)
    |> tap(fn _ -> MLIR.Module.destroy(pdl_mod) end)
    |> tap(fn _ -> MLIR.FrozenRewritePatternSet.destroy(frozen_set) end)
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
      Beaver.MLIR.Diagnostic,
      Beaver.MLIR.StringRef
    ] do
  defimpl String.Chars, for: m do
    defdelegate to_string(mlir), to: Beaver.MLIR
  end
end
