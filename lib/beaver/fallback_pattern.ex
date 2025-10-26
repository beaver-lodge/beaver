defmodule Beaver.FallbackPattern do
  @behaviour Beaver.MLIR.RewritePattern
  @moduledoc false
  def construct(state), do: state
  def destruct(_state), do: :ok
  def match_and_rewrite(_pattern, _op, _rewriter, state), do: {:ok, state}
end
