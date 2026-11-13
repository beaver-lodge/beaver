#ifndef MLIR_C_BEAVER_REMARKS_H
#define MLIR_C_BEAVER_REMARKS_H
#include "mlir-c/IR.h"
#ifdef __cplusplus
extern "C" {
#endif
// Copies each callback-scoped remark into JSON. Takes ownership of userData even on failure; the engine releases it on
// finalization or context teardown.
MLIR_CAPI_EXPORTED bool beaverContextEnableOptimizationRemarks(
    MlirContext context, MlirStringRef filters, bool finalPolicy,
    MlirStringCallback callback, void *userData, void (*deleteUserData)(void *));
MLIR_CAPI_EXPORTED void beaverContextFinalizeOptimizationRemarks(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverContextHasOptimizationRemarks(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverEmitOptimizationRemark(MlirLocation location, MlirStringRef payload);
#ifdef __cplusplus
}
#endif
#endif
