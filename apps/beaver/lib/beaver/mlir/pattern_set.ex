defmodule Beaver.MLIR.PatternSet do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def get(opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.beaverRewritePatternSetGet(ctx)
  end

  def insert(pattern_set, %Beaver.MLIR.CAPI.MlirModule{} = module) do
    pattern_module = CAPI.beaverPDLPatternGet(module)
    insert(pattern_set, pattern_module)
  end

  def insert(pattern_set, %Beaver.MLIR.CAPI.MlirPDLPatternModule{} = pattern_module) do
    CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pattern_module)
    pattern_set
  end

  def apply!(region = %CAPI.MlirRegion{}, pattern_set) do
    result = CAPI.beaverApplyOwnedPatternSetOnRegion(region, pattern_set)
    if not MLIR.LogicalResult.success?(result), do: raise("fail to apply patterns")
    region
  end

  def apply!(operation = %CAPI.MlirOperation{}, pattern_set) do
    result = CAPI.beaverApplyOwnedPatternSetOnOperation(operation, pattern_set)
    if not MLIR.LogicalResult.success?(result), do: raise("fail to apply patterns")
    operation
  end

  def apply!(module = %CAPI.MlirModule{}, pattern_set) do
    MLIR.Operation.from_module(module) |> apply!(pattern_set)
  end
end
