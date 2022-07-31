defmodule Manx.Compiler do
  @behaviour Nx.Defn.Compiler
  @impl true
  defdelegate __jit__(key, vars, fun, args, opts), to: Manx.Defn
end
