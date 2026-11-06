defmodule Beaver.MLIR.PostDominanceInfo do
  @moduledoc "MLIR analysis handle; use the corresponding functions in Beaver.MLIR.CAPI."
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native
end
