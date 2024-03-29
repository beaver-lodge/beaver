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
  Diagnostic,
  DiagnosticSeverity,
  PassManager,
  RewritePatternSet,
  OperationState,
  SymbolTable,
  RegisteredOperationName,
  ExternalPassCallbacks,
  OpPassManager,
  AsmState,
  OperationWalkCallback,
  WalkOrder,
  BytecodeWriterConfig,
  OpPrintingFlags,
  LLVMThreadPool,
  TypeIDAllocator
]

for m <- mlir_mods do
  m = Module.concat(Beaver.MLIR, m)

  defmodule m do
    use Kinda.ResourceKind,
      root_module: Beaver.MLIR.CAPI,
      forward_module: Beaver.Native
  end
end
