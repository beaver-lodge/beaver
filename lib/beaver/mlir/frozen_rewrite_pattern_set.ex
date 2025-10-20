defmodule Beaver.MLIR.FrozenRewritePatternSet do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native
  alias Beaver.MLIR
  defdelegate destroy(set), to: MLIR.CAPI, as: :mlirFrozenRewritePatternSetDestroy
  defdelegate destroy(ctx, set), to: MLIR.RewritePatternSet, as: :destroy
end
