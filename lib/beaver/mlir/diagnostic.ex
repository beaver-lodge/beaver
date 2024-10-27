defmodule Beaver.MLIR.Diagnostic do
  alias Beaver.MLIR
  use Kinda.ResourceKind, forward_module: Beaver.Native

  def severity(%__MODULE__{} = diagnostic) do
    i = MLIR.CAPI.mlirDiagnosticGetSeverity(diagnostic) |> Beaver.Native.to_term()
    Enum.at([:error, :warning, :note, :remark], i) || raise "unknown severity: #{i}"
  end

  defdelegate detach(ctx, handler_id), to: MLIR.CAPI, as: :mlirContextDetachDiagnosticHandler
end
