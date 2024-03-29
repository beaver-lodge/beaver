defmodule Beaver.MLIR.Context do
  @moduledoc """
  This module defines functions creating or destroying MLIR context.
  """
  alias Beaver.MLIR

  use Kinda.ResourceKind, forward_module: Beaver.Native

  @type context_option :: {:allow_unregistered, boolean()}
  @spec create(context_option()) :: __MODULE__.t()
  @default_context_option [allow_unregistered: false]
  @doc """
  create a MLIR context and register all dialects
  """
  def create(opts \\ @default_context_option) do
    allow_unregistered = opts[:allow_unregistered] || @default_context_option[:allow_unregistered]
    ctx = %__MODULE__{ref: MLIR.CAPI.beaver_raw_get_context_load_all_dialects()}
    Beaver.Exterior.register_all(ctx)
    # TODO: do not load dialects twice
    MLIR.CAPI.mlirContextLoadAllAvailableDialects(ctx)

    MLIR.CAPI.mlirContextSetAllowUnregisteredDialects(
      ctx,
      Beaver.Native.Bool.make(allow_unregistered)
    )

    ctx
  end

  defdelegate destroy(ctx), to: MLIR.CAPI, as: :mlirContextDestroy
end
