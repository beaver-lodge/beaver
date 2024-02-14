defmodule Beaver.Diagnostic do
  alias Beaver.MLIR

  def attach(%MLIR.Context{ref: ctx_ref}, handler \\ :stderr)
      when is_pid(handler) or handler == :stderr do
    MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx_ref, handler)
    |> Beaver.Native.check!()
  end

  defdelegate detach(ctx, handler_id), to: MLIR.CAPI, as: :mlirContextDetachDiagnosticHandler
end
