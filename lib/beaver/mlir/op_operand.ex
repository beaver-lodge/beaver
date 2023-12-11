defmodule Beaver.MLIR.OpOperand do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind,
    forward_module: Beaver.Native
end
