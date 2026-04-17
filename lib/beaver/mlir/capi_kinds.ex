mlir_mods = [
  ShapedTypeComponentsCallback,
  TypeID,
  TypesCallback,
  IntegerSet,
  AffineExpr,
  StringCallback,
  DialectHandle,
  SparseTensorLevelType,
  DialectRegistry,
  DiagnosticHandlerID,
  DiagnosticHandler,
  DiagnosticSeverity,
  ExternalPassCallbacks,
  ConversionTarget,
  ConversionPattern,
  ConversionPatternRewriter,
  TypeConverter,
  ConversionConfig,
  OpPassManager,
  AsmState,
  OperationWalkCallback,
  WalkOrder,
  BytecodeWriterConfig,
  OpPrintingFlags,
  LLVMRawFdOStream,
  LLVMThreadPool,
  TypeIDAllocator,
  DynamicOpTrait,
  DynamicOpTraitCallbacks,
  DynamicTypeDefinition,
  DynamicAttrDefinition,
  MemoryEffectInstancesList,
  MemoryEffectsOpInterfaceCallbacks,
  TransformResults,
  TransformRewriter,
  TransformState,
  TransformOpInterfaceCallbacks,
  PatternDescriptorOpInterfaceCallbacks,
  TransformOptions
]

for m <- mlir_mods do
  m = Module.concat(Beaver.MLIR, m)

  defmodule m do
    use Kinda.ResourceKind, forward_module: Beaver.Native
  end
end
