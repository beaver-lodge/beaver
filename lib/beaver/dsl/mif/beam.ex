defmodule Beaver.MIF.BEAM do
  defmacro handle_intrinsic(:env, [], _opts) do
    quote do
      var!(mif_internal_beam_env)
    end
  end
end
