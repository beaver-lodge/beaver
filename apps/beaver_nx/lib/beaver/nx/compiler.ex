defmodule Beaver.Nx.Compiler do
  @behaviour Nx.Defn.Compiler
  @impl true
  defdelegate __jit__(key, vars, fun, args, opts), to: Beaver.Nx.Defn
end
