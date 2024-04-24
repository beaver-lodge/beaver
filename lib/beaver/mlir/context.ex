defmodule Beaver.MLIR.Context do
  @moduledoc """
  This module defines functions creating or destroying MLIR context.
  """
  alias Beaver.MLIR
  import MLIR.CAPI
  use Kinda.ResourceKind, forward_module: Beaver.Native

  defp append_all_dialects(ctx) do
    registry = mlirDialectRegistryCreate()
    mlirRegisterAllDialects(registry)
    mlirContextAppendDialectRegistry(ctx, registry)
    mlirDialectRegistryDestroy(registry)
  end

  @type context_option :: {:allow_unregistered, boolean()} | {:all_dialects, boolean()}
  @spec create(context_option()) :: __MODULE__.t()
  @default_context_option [allow_unregistered: false, all_dialects: true]
  @doc """
  create a MLIR context, it registers all dialects by default
  """
  def create(opts \\ @default_context_option) do
    allow_unregistered = opts[:allow_unregistered] || @default_context_option[:allow_unregistered]
    all_dialects = opts[:all_dialects] || @default_context_option[:all_dialects]
    ctx = mlirContextCreate()

    if all_dialects do
      append_all_dialects(ctx)
    end

    mlirContextSetAllowUnregisteredDialects(ctx, Beaver.Native.Bool.make(allow_unregistered))
    Beaver.Exterior.register_all(ctx)
    # TODO: do not load dialects twice
    mlirContextLoadAllAvailableDialects(ctx)
    ctx
  end

  defdelegate destroy(ctx), to: MLIR.CAPI, as: :mlirContextDestroy
end
