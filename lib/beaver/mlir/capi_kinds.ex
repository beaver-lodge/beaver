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
  RewritePatternSet,
  SymbolTable,
  ExternalPassCallbacks,
  OpPassManager,
  AsmState,
  OperationWalkCallback,
  WalkOrder,
  BytecodeWriterConfig,
  OpPrintingFlags,
  LLVMThreadPool,
  TypeIDAllocator,
  TransformOptions
]

for m <- mlir_mods do
  m = Module.concat(Beaver.MLIR, m)

  defmodule m do
    use Kinda.ResourceKind, forward_module: Beaver.Native
  end
end
