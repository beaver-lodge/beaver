mlir_mods = [
  MlirShapedTypeComponentsCallback,
  MlirTypeID,
  MlirTypesCallback,
  MlirIntegerSet,
  MlirAffineExpr,
  MlirStringCallback,
  MlirDialectHandle,
  Enum_MlirSparseTensorLevelType,
  MlirDialectRegistry,
  MlirDiagnosticHandlerID,
  MlirDiagnosticHandler,
  MlirDiagnostic,
  MlirDiagnosticSeverity,
  MlirPassManager,
  MlirRewritePatternSet,
  MlirExecutionEngine,
  MlirOperationState,
  MlirSymbolTable,
  MlirRegisteredOperationName,
  MlirExternalPass,
  MlirExternalPassCallbacks,
  MlirOpPassManager,
  MlirAsmState,
  MlirOperationWalkCallback,
  MlirWalkOrder,
  MlirBytecodeWriterConfig,
  MlirOpPrintingFlags,
  MlirLlvmThreadPool,
  MlirTypeIDAllocator
]

for m <- mlir_mods do
  m = Module.concat(Beaver.MLIR.CAPI, m)

  defmodule m do
    use Kinda.ResourceKind,
      root_module: Beaver.MLIR.CAPI,
      forward_module: Beaver.Native
  end
end
