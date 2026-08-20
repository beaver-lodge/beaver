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
  TransformResults,
  TransformRewriter,
  TransformState,
  TransformOpInterfaceCallbacks,
  PatternDescriptorOpInterfaceCallbacks,
  TransformOptions,
  ConditionallySpeculatableOpInterfaceCallbacks,
  MemoryEffect,
  MemoryEffectInstance,
  OpOperandReplaceFilterCallback,
  RewriterBaseInsertPoint,
  SideEffectResource,
  TypeConverter1ToNConversionCallback,
  TypeConverter1ToNTargetMaterializationCallback,
  TypeConverterConversionResults,
  TypeConverterSourceMaterializationCallback,
  TypeConverterTargetMaterializationCallback
]

for m <- mlir_mods do
  m = Module.concat(Beaver.MLIR, m)

  defmodule m do
    use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native
  end
end
