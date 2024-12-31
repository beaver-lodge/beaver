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
    {str, _} =
      for d <- diagnostics,
          reduce: {prefix, 0} do
        acc ->
          MLIR.Diagnostic.walk(d, acc, fn {_, loc, d, _}, {str, level} ->
            prefix = String.duplicate(" ", level * 2)
            d = String.replace(d, ~r"\n", "\n#{prefix}  ")
            {"#{str}\n#{prefix}at #{loc}: #{d}", level + 1}
          end)
      end

    str
  end
end
