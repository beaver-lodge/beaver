defmodule Beaver.MLIR.Dialect.Arith do
  alias Beaver.MLIR
  import MLIR.Sigils

  def constant(value, type) do
  end

  def constant(true) do
    ~t{i1}
  end
end
