defmodule Beaver.MLIR.LogicalResult do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  def success?(result) do
    result
    |> MLIR.CAPI.beaverLogicalResultIsSuccess()
    |> Beaver.Native.to_term()
  end
end
