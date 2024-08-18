defmodule Beaver.MLIR.PatternSet do
  @moduledoc false
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def apply!(container, pattern_set) do
    case apply_(container, pattern_set) do
      {:ok, container} ->
        container

      _ ->
        raise "failed to apply pattern set"
    end
  end

  defp do_apply(container, pattern_set, driver) when is_function(driver, 2) do
    result = driver.(container, pattern_set)

    if MLIR.LogicalResult.success?(result) do
      {:ok, container}
    else
      :error
    end
  end

  def apply_(%MLIR.Module{} = module, pattern_set) do
    do_apply(module, pattern_set, &CAPI.beaverApplyPatternsAndFoldGreedily/2)
  end
end
