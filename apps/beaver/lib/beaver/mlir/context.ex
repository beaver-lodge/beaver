defmodule Beaver.MLIR.Context do
  alias Beaver.MLIR
  require MLIR.CAPI

  @doc """
  create a MLIR context and register all dialects
  """
  def create(allow_unregistered: allow_unregistered) do
    ctx = %MLIR.CAPI.MlirContext{ref: MLIR.CAPI.beaver_raw_get_context_load_all_dialects()}
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
end
