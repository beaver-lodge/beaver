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
    MlirConversionPatternRewriter rewriter, void *userData,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  (void)host;
  (void)operation;
  (void)nOperands;
  (void)operands;
  (void)userData;
  (void)diagnostic;
  (void)diagnosticUserData;

  host->eraseOperation(rewriter, operation);
  return mlirLogicalResultSuccess();
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
      !host->addPattern || !host->eraseOperation)
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelPattern pattern = {
      sizeof(MlirBeaverCompilerKernelPattern),
      mlirStringRefCreate("fixture.noop", sizeof("fixture.noop") - 1),
      mlirStringRefCreate("fixture.noop", sizeof("fixture.noop") - 1),
      mlirStringRefCreate("1", sizeof("1") - 1),
      1,
      fixture_rewrite,
      NULL,
      NULL,
  };

  return host->addPattern(hostContext, patterns, typeConverter, &pattern);
}
