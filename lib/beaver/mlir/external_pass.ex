defmodule Beaver.MLIR.ExternalPass do
  @moduledoc false
  # Lower level API to work with MLIR CAPI's external pass (pass defined in C).
  # Note that external pass is a C API specific concept, not a concept generally available in MLIR, so we don't expose it from Elixir.
  use Kinda.ResourceKind, forward_module: Beaver.Native
end
