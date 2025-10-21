defmodule Beaver.MLIR.PatternRewriter do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native
  alias Beaver.MLIR

  defdelegate as_base(pattern), to: MLIR.CAPI, as: :mlirPatternRewriterAsBase
end
