#ifndef APPS_BEAVER_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PASS_H_
#define APPS_BEAVER_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PASS_H_

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetArgument(MlirPass pass);
MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetDescription(MlirPass pass);
MLIR_CAPI_EXPORTED bool beaverIsOpNameTerminator(MlirStringRef op_name,
                                                 MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PASS_H_
