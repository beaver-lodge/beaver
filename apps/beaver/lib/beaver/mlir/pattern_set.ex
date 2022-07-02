defmodule Beaver.MLIR.PatternSet do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def get(ctx, %Beaver.MLIR.CAPI.MlirModule{} = module) do
    pattern_set = CAPI.beaverRewritePatternSetGet(ctx)
    module = CAPI.beaverPDLPatternGet(module)
    CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, module)
    pattern_set
  end

  def insert(pattern_set, pattern) do
    # TODO: support extracting mlir context from pattern_set so it could support create from string
    CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pattern)
  end

  def apply!(module, pattern_set) do
    region = CAPI.mlirOperationGetFirstRegion(module)
    result = CAPI.beaverApplyOwnedPatternSet(region, pattern_set)
    if not MLIR.LogicalResult.success?(result), do: raise("fail to apply patterns")
  end
end
