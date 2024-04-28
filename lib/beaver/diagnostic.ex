defmodule Beaver.Diagnostic do
  alias Beaver.MLIR

  def attach(%MLIR.Context{ref: ctx_ref}, handler \\ :stderr)
      when is_pid(handler) or handler == :stderr do
    MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx_ref, handler)
  end

  def callback() do
    %MLIR.StringCallback{ref: MLIR.CAPI.beaver_raw_get_diagnostic_string_callback()}
  end

  defdelegate detach(ctx, handler_id), to: MLIR.CAPI, as: :mlirContextDetachDiagnosticHandler
end
