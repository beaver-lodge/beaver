defmodule Beaver.MLIR.Module do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @doc """
  Create a MLIR module by parsing string.
  """
  def create(str, opts \\ []) when is_binary(str) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        {module, diagnostics} =
          CAPI.mlirModuleCreateParseWithDiagnostics(ctx, ctx, MLIR.StringRef.create(str))

        if MLIR.null?(module) do
          {:error, diagnostics}
        else
          {:ok, module}
        end
      end
    )
  end

  def create!(str, opts \\ []) when is_binary(str) do
    res =
      Beaver.Deferred.from_opts(opts, fn ctx -> Beaver.Deferred.create(create(str, opts), ctx) end)

    case res do
      f when is_function(f, 1) ->
        raise ArgumentError, "calling a bang function to parse module must be eager"

      {:error, diagnostics} ->
        raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "fail to parse module")

      {:ok, module} ->
        MLIR.verify!(module)
    end
  end

  use Kinda.ResourceKind, forward_module: Beaver.Native

  defdelegate destroy(module), to: CAPI, as: :mlirModuleDestroy

  @doc """
  Gets the body of the module, i.e. the only block it contains.
  """
  defdelegate body(module), to: CAPI, as: :mlirModuleGetBody
  defdelegate from_operation(op), to: CAPI, as: :mlirModuleFromOperation
  defdelegate empty(location), to: CAPI, as: :mlirModuleCreateEmpty
end
