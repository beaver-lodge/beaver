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
        CAPI.mlirModuleCreateParse(ctx, MLIR.StringRef.create(str))
      end
    )
  end

  def create!(str, opts \\ []) when is_binary(str) do
    create(str, opts) |> verify!()
  end

  use Kinda.ResourceKind, forward_module: Beaver.Native

  def verify!(module) do
    if MLIR.null?(module) do
      raise "module is null"
    end

    MLIR.verify!(module)
  end

  defdelegate destroy(module), to: CAPI, as: :mlirModuleDestroy
  defdelegate body(module), to: CAPI, as: :mlirModuleGetBody
  defdelegate from_operation(op), to: CAPI, as: :mlirModuleFromOperation
  defdelegate empty(location), to: CAPI, as: :mlirModuleCreateEmpty
end
