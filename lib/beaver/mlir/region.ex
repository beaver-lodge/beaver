defmodule Beaver.MLIR.Region do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native

  def under(region, f) when is_function(f, 0) do
    f.()
    region
  end

  defdelegate append(region, block), to: CAPI, as: :mlirRegionAppendOwnedBlock
  defdelegate insert(region, index, block), to: CAPI, as: :mlirRegionInsertOwnedBlock
end
