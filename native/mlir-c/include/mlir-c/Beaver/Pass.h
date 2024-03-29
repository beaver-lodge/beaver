#ifndef APPS_BEAVER_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PASS_H_
#define APPS_BEAVER_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PASS_H_

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetArgument(MlirPass pass);
MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetName(MlirPass pass);
MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetDescription(MlirPass pass);
MLIR_CAPI_EXPORTED MlirContext
beaverPassManagerGetContext(MlirPassManager passManager);
MLIR_CAPI_EXPORTED void
beaverPassManagerEnableTiming(MlirPassManager passManager);

#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PASS_H_
