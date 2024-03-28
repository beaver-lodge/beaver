defmodule Beaver.MIF.Env do
  use Beaver

  def handle_intrinsic(:t, [], opts) do
    Beaver.ENIF.mlir_t(:env, opts)
  end
end
