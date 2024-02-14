defmodule Beaver.MLIR.Context do
  @moduledoc """
  This module defines functions creating or destroying MLIR context.
  """
  alias Beaver.MLIR
  require MLIR.CAPI

  use Kinda.ResourceKind,
    forward_module: Beaver.Native,
    fields: [diagnostic_server: nil]

  @doc """
  create a MLIR context and register all dialects
  """
  @type context_option ::
          {:allow_unregistered, boolean()} | {:diagnostic_server, pid() | :default | nil}
  @spec create(context_option()) :: MLIR.Context.t()
  def create(opts) do
    allow_unregistered = opts[:allow_unregistered] || false
    diagnostic_server = opts[:diagnostic_server] || nil

    ctx = %__MODULE__{
      ref: MLIR.CAPI.beaver_raw_get_context_load_all_dialects(),
      diagnostic_server: diagnostic_server
    }

    MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx.ref, diagnostic_server)
    |> Beaver.Native.check!()

    Beaver.Exterior.register_all(ctx)
    # TODO: do not load dialects twice
    MLIR.CAPI.mlirContextLoadAllAvailableDialects(ctx)

    MLIR.CAPI.mlirContextSetAllowUnregisteredDialects(
      ctx,
      Beaver.Native.Bool.make(allow_unregistered)
    )

    ctx
  end

  def create() do
    create(allow_unregistered: false)
  end

  def destroy(%__MODULE__{diagnostic_server: diagnostic_server} = ctx) do
    if diagnostic_server do
      GenServer.stop(diagnostic_server)
    end

    MLIR.CAPI.mlirContextDestroy(ctx)
  end
end
