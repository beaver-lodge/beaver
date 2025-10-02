#ifndef MLIR_C_BEAVER_DEBUG_H_
#define MLIR_C_BEAVER_DEBUG_H_

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void beaverSetGlobalDebugTypes(const MlirStringRef *types,
                                                  intptr_t n);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_DEBUG_H_
