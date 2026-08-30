#include "mlir-c/Beaver/CompilerKernel.h"

#ifndef FIXTURE_ABI_VERSION
#define FIXTURE_ABI_VERSION MLIR_BEAVER_COMPILER_KERNEL_ABI_VERSION
#endif

#ifndef FIXTURE_IDENTITY
#error "FIXTURE_IDENTITY must be defined"
#endif

#if defined(_WIN32)
#define FIXTURE_EXPORT __declspec(dllexport)
#else
#define FIXTURE_EXPORT __attribute__((visibility("default")))
#endif

static MlirLogicalResult fixture_rewrite(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, MlirTypeConverter typeConverter,
    void *userData,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  (void)host;
  (void)operation;
  (void)userData;

  if (nOperands != 2) {
    static const char message[] = "fixture.add expects two converted operands";
    diagnostic(mlirStringRefCreate(message, sizeof(message) - 1),
               diagnosticUserData);
    return mlirLogicalResultFailure();
  }

  MlirValue sourceResult;
  MlirType sourceType;
  MlirType convertedType;
  MlirLocation location;
#if defined(FIXTURE_BAD_RESULT_INDEX)
  const intptr_t sourceResultIndex = 1;
#else
  const intptr_t sourceResultIndex = 0;
#endif
  if (mlirLogicalResultIsFailure(host->operationResult(
          operation, sourceResultIndex, &sourceResult, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationLocation(
          operation, &location, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->valueType(
          sourceResult, &sourceType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->convertType(
          typeConverter, sourceType, &convertedType,
          diagnostic, diagnosticUserData)))
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelOperation descriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("arith.addi", sizeof("arith.addi") - 1),
      location,
      nOperands,
      operands,
      1,
      &convertedType,
      0,
      NULL,
  };

  MlirOperation replacement;
  MlirValue replacementResult;
  if (mlirLogicalResultIsFailure(host->createOperation(
          rewriter, &descriptor, &replacement, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationResult(
          replacement, 0, &replacementResult, diagnostic,
          diagnosticUserData)))
    return mlirLogicalResultFailure();

#if defined(FIXTURE_PARTIAL_FAILURE)
  static const char message[] = "fixture requested rollback after operation creation";
  diagnostic(mlirStringRefCreate(message, sizeof(message) - 1),
             diagnosticUserData);
  return mlirLogicalResultFailure();
#endif

  return host->replaceOperationWithValues(
      rewriter, operation, 1, &replacementResult, diagnostic,
      diagnosticUserData);
}

FIXTURE_EXPORT uint32_t fixture_abi_version(void) {
  return FIXTURE_ABI_VERSION;
}

FIXTURE_EXPORT MlirStringRef fixture_manifest(void) {
  static const char identity[] = FIXTURE_IDENTITY;
  return mlirStringRefCreate(identity, sizeof(identity) - 1);
}

FIXTURE_EXPORT MlirLogicalResult fixture_populate(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter,
    const MlirBeaverCompilerKernelHostAPI *host, void *hostContext,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  (void)diagnostic;
  (void)diagnosticUserData;

  if (!host || host->abiVersion != MLIR_BEAVER_COMPILER_KERNEL_ABI_VERSION ||
      host->structSize < sizeof(MlirBeaverCompilerKernelHostAPI) ||
      !host->addPattern || !host->operationResult ||
      !host->operationLocation || !host->valueType || !host->convertType ||
      !host->createOperation || !host->replaceOperationWithValues ||
      !host->eraseOperation)
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelPattern pattern = {
      sizeof(MlirBeaverCompilerKernelPattern),
      mlirStringRefCreate("fixture.add", sizeof("fixture.add") - 1),
      mlirStringRefCreate("fixture.add", sizeof("fixture.add") - 1),
      mlirStringRefCreate("1", sizeof("1") - 1),
      1,
      fixture_rewrite,
      NULL,
      NULL,
  };

  return host->addPattern(hostContext, patterns, typeConverter, &pattern);
}
