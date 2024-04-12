defmodule Beaver.MIF.Term do
  use Beaver.MIF.Intrinsic

  defi t(opts) do
    Beaver.ENIF.mlir_t(:term, opts)
  end
end
