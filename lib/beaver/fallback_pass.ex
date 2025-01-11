defmodule Beaver.FallbackPass do
  @behaviour Beaver.MLIR.Pass
  @moduledoc false
  def destruct(_), do: :ok
  def initialize(_ctx, state), do: {:ok, state}
  def clone(state), do: state
  def run(_, state), do: state
end
