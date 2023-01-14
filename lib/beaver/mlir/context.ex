defmodule Beaver.MLIR.Context do
  @moduledoc """
  This module defines functions creating or destroying MLIR context.
  """
  alias Beaver.MLIR
  require MLIR.CAPI

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  @doc """
  create a MLIR context and register all dialects
  """
  def create(allow_unregistered: allow_unregistered) do
    ctx = %__MODULE__{ref: MLIR.CAPI.beaver_raw_get_context_load_all_dialects()}
    MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx.ref) |> Beaver.Native.check!()

    MLIR.CAPI.mlirContextSetAllowUnregisteredDialects(
      ctx,
      Beaver.Native.Bool.make(allow_unregistered)
    )

    ctx
  end

  def create() do
    create(allow_unregistered: false)
  end

  defdelegate destroy(ctx), to: MLIR.CAPI, as: :mlirContextDestroy
end
