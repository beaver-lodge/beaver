defmodule Beaver.MLIR.PatternSet do
  @moduledoc false
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  def get(ctx) do
    CAPI.beaverRewritePatternSetGet(ctx)
  end

  def insert(pattern_set, %Beaver.MLIR.Module{} = module) do
    pattern_module = CAPI.beaverPDLPatternGet(module)
    insert(pattern_set, pattern_module)
  end

  def insert(pattern_set, %Beaver.MLIR.CAPI.MlirPDLPatternModule{} = pattern_module) do
    CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pattern_module)
    pattern_set
  end

  def apply!(container, pattern_set) do
    with {:ok, container} <- apply_(container, pattern_set) do
      container
    else
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

  def apply_(%MLIR.Region{} = region, pattern_set) do
    do_apply(region, pattern_set, &CAPI.beaverApplyOwnedPatternSetOnRegion/2)
  end

  def apply_(%MLIR.Operation{} = operation, pattern_set) do
    do_apply(operation, pattern_set, &CAPI.beaverApplyOwnedPatternSetOnOperation/2)
  end

  def apply_(%MLIR.Module{} = module, pattern_set) do
    with {:ok, %MLIR.Operation{}} <- MLIR.Operation.from_module(module) |> apply_(pattern_set) do
      {:ok, module}
    else
      _ ->
        :error
    end
  end
end
