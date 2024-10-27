defmodule Beaver.Diagnostic do
  alias Beaver.MLIR

  def attach(%MLIR.Context{ref: ctx_ref}, handler) when is_pid(handler) do
    MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx_ref, handler)
  end

  defdelegate detach(ctx, handler_id), to: MLIR.CAPI, as: :mlirContextDetachDiagnosticHandler
end
