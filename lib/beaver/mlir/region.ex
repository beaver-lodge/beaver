defmodule Beaver.MLIR.Region do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  def under(region, f) when is_function(f, 0) do
    f.()
    region
  end
end
