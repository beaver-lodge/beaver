defmodule Beaver.MLIR.Module do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def create(context, str) when is_binary(str) do
    CAPI.mlirModuleCreateParse(context, MLIR.StringRef.create(str))
  end

  def create!(context, str) when is_binary(str) do
    module = create(context, str)
    verify!(module)
    module
  end

  use Kinda.ResourceKind, forward_module: Beaver.Native

  def is_null(module) do
    CAPI.beaverModuleIsNull(module) |> Beaver.Native.to_term()
  end

  defp not_null!(module) do
    if is_null(module) do
      raise "module is null"
    end
  end

  def verify!(module) do
    not_null!(module)
    MLIR.Operation.verify!(module)
  end

  def destroy(module) do
    CAPI.mlirModuleDestroy(module)
  end

  defdelegate body(module), to: CAPI, as: :mlirModuleGetBody
end
