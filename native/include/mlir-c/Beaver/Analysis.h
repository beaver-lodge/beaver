#ifndef MLIR_C_BEAVER_ANALYSIS_H
#define MLIR_C_BEAVER_ANALYSIS_H
#include "mlir-c/Interfaces.h"
#include "mlir-c/Rewrite.h"
#ifdef __cplusplus
extern "C" {
#endif
// Internal adapters. The BEAM resource owns the session; no analysis handles
// or operation-specific property pointers escape to Elixir.
MLIR_CAPI_EXPORTED void *beaverAnalysisCreate(MlirOperation root);
MLIR_CAPI_EXPORTED void beaverAnalysisDestroy(void *session);
MLIR_CAPI_EXPORTED void beaverAnalysisInvalidate(void *session, unsigned preserved);
MLIR_CAPI_EXPORTED uint64_t beaverAnalysisGeneration(void *session);
MLIR_CAPI_EXPORTED unsigned beaverAnalysisCached(void *session);
MLIR_CAPI_EXPORTED int beaverAnalysisQuery(void *session, unsigned kind, MlirOperation lhs, MlirOperation rhs, MlirValue value);
MLIR_CAPI_EXPORTED bool beaverAnalysisLayout(void *session, MlirOperation scope, MlirType type, uint64_t *result);

MLIR_CAPI_EXPORTED MlirLogicalResult beaverInferReturnTypes(
    MlirStringRef name, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    MlirAttribute properties, intptr_t nRegions, MlirRegion *regions,
    MlirTypesCallback callback, void *data);
MLIR_CAPI_EXPORTED MlirLogicalResult beaverInferReturnComponents(
    MlirStringRef name, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    MlirAttribute properties, intptr_t nRegions, MlirRegion *regions,
    MlirShapedTypeComponentsCallback callback, void *data);
MLIR_CAPI_EXPORTED MlirLogicalResult beaverConversionRun(
    unsigned mode, MlirOperation root, MlirConversionTarget target,
    MlirFrozenRewritePatternSet patterns, MlirConversionConfig config,
    MlirStringCallback report, void *data);
#ifdef __cplusplus
}
#endif
#endif
