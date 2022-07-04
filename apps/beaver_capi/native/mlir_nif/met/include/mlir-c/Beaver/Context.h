#ifndef APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_
#define APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void beaverEnterMultiThreadedExecution(MlirContext context);
MLIR_CAPI_EXPORTED void beaverExitMultiThreadedExecution(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_
