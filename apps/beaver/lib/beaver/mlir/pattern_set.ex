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

  def apply!(module, pattern_set) do
    region = CAPI.mlirOperationGetFirstRegion(module)
    result = CAPI.beaverApplyOwnedPatternSet(region, pattern_set)
    if not MLIR.LogicalResult.success?(result), do: raise("fail to apply patterns")
    module
  end
end
