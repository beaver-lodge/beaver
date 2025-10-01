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
  def severity(i) when is_integer(i) do
    Enum.at(@mlir_severity_levels, i) || :unknown
  end

  def severity(%__MODULE__{} = diagnostic) do
    MLIR.CAPI.mlirDiagnosticGetSeverity(diagnostic) |> Beaver.Native.to_term() |> severity()
  end

  defdelegate detach(ctx, handler_id), to: MLIR.CAPI, as: :mlirContextDetachDiagnosticHandler

  def walk({_, _, _, []} = diagnostic, acc, fun) do
    fun.(diagnostic, acc)
  end

  def walk({_, _, _, nested} = diagnostic, acc, fun) do
    acc = fun.(diagnostic, acc)

    Enum.reduce(nested, acc, fn d, acc ->
      walk(d, acc, fun)
    end)
  end

  def format(diagnostics, prefix \\ "") do
    diagnostics
    |> Enum.reduce(prefix, fn diagnostic, acc ->
      {str, _} = walk(diagnostic, {acc, 0}, &format_diagnostic/2)
      str
    end)
  end

  defp format_diagnostic({_, loc, message, _}, {str, level}) do
    indent = String.duplicate(" ", level * 2)
    formatted_message = String.replace(message, ~r"\n", "\n#{indent}  ")
    {"#{str}\n#{indent}at #{loc}: #{formatted_message}", level + 1}
  end

  def emit(ctx, msg) do
    msg_str = MLIR.StringRef.create(msg) |> MLIR.StringRef.data()
    MLIR.CAPI.mlirEmitError(ctx, msg_str)
  end
end
