defmodule Beaver.MLIR.Diagnostic do
  @moduledoc """
  This module provides functions to work with MLIR diagnostics.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native
  alias Beaver.MLIR

  @mlir_severity_levels [:error, :warning, :note, :remark]
  @doc """
  Returns the severity level of a diagnostic.

  If the MLIR severity level is not recognized, it returns `:unknown`.
  """
  def severity(%__MODULE__{} = diagnostic) do
    i = MLIR.CAPI.mlirDiagnosticGetSeverity(diagnostic) |> Beaver.Native.to_term()
    Enum.at(@mlir_severity_levels, i) || :unknown
  end

  defdelegate detach(ctx, handler_id), to: MLIR.CAPI, as: :mlirContextDetachDiagnosticHandler
end
