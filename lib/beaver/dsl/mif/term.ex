defmodule Beaver.MIF.Term do
  use Beaver

  def handle_intrinsic(:t, [], opts) do
    Beaver.ENIF.mlir_t(:term, opts)
  end
end
