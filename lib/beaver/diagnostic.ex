defmodule Beaver.Diagnostic do
  alias Beaver.MLIR

  def attach(%MLIR.Context{} = ctx, handler \\ :stderr)
      when is_pid(handler) or handler == :stderr do
    ref =
      MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx.ref, handler)
      |> Beaver.Native.check!()

    %MLIR.CAPI.MlirDiagnosticHandlerID{ref: ref}
  end

  def detach(%MLIR.Context{} = ctx, handler_id) do
    MLIR.CAPI.mlirContextDetachDiagnosticHandler(ctx, handler_id)
  end
end
