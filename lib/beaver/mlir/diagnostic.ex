defmodule Beaver.MLIR.Diagnostic do
  alias Beaver.MLIR
  use Kinda.ResourceKind, forward_module: Beaver.Native

  @severity_levels [:error, :warning, :note, :remark, :unknown]
  @doc """
  Returns the severity level of a diagnostic.

  If the MLIR severity level is not recognized, it returns `:unknown`.
  """
  def severity(%__MODULE__{} = diagnostic) do
    i = MLIR.CAPI.mlirDiagnosticGetSeverity(diagnostic) |> Beaver.Native.to_term()
    Enum.at(@severity_levels, i) || :unknown
  end

  defdelegate detach(ctx, handler_id), to: MLIR.CAPI, as: :mlirContextDetachDiagnosticHandler
end
