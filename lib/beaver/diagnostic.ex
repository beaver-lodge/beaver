defmodule Beaver.Diagnostic do
  alias Beaver.MLIR

  def attach(%MLIR.Context{ref: ctx_ref}, handler \\ :stderr)
      when is_pid(handler) or handler == :stderr do
    MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx_ref, handler)
    |> Beaver.Native.check!()
  end

  def detach(%MLIR.Context{} = ctx, handler_id) do
    MLIR.CAPI.mlirContextDetachDiagnosticHandler(ctx, handler_id)
  end
end
