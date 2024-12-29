defmodule Beaver.MLIR.Context do
  @moduledoc """
  This module defines functions creating or destroying MLIR context.
  """
  alias Beaver.MLIR
  import MLIR.CAPI
  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc """
  Run a function with a registry appended to the context.
  """
  def with_registry(ctx, fun) when is_function(fun, 1) do
    registry = mlirDialectRegistryCreate()
    mlirContextAppendDialectRegistry(ctx, registry)

    try do
      fun.(registry)
    after
      mlirDialectRegistryDestroy(registry)
    end
  end

  # create an interim registry and append all dialects to the context
  defp load_all_dialects(ctx) do
    with_registry(ctx, fn registry ->
      mlirRegisterAllDialects(registry)
      mlirContextAppendDialectRegistry(ctx, registry)
      mlirContextLoadAllAvailableDialects(ctx)
    end)
  end

  @type context_option :: {:allow_unregistered, boolean()} | {:all_dialects, boolean()}
  @spec create([context_option()]) :: __MODULE__.t()
  @doc """
  Create a MLIR context. By default it registers and loads all dialects.
  """
  def create(opts \\ []) do
    allow_unregistered = Keyword.get(opts, :allow_unregistered, false)
    all_dialects = Keyword.get(opts, :all_dialects, true)

    mlirContextCreate()
    |> tap(fn ctx -> if all_dialects, do: load_all_dialects(ctx) end)
    |> tap(&mlirContextSetAllowUnregisteredDialects(&1, allow_unregistered))
  end

  defdelegate destroy(ctx), to: MLIR.CAPI, as: :mlirContextDestroy
  defdelegate register_translations(ctx), to: MLIR.CAPI, as: :mlirRegisterAllLLVMTranslations
end
