#ifndef APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
#define APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirRegisteredOperationName, void);

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED bool beaverIsOpNameTerminator(MlirStringRef op_name,
                                                 MlirContext context);

MLIR_CAPI_EXPORTED intptr_t
beaverGetNumRegisteredOperations(MlirContext context);

MLIR_CAPI_EXPORTED MlirRegisteredOperationName
beaverGetRegisteredOperationName(MlirContext context, intptr_t pos);

MLIR_CAPI_EXPORTED MlirStringRef
beaverRegisteredOperationNameGetDialectName(MlirRegisteredOperationName name);

MLIR_CAPI_EXPORTED MlirStringRef
beaverRegisteredOperationNameGetOpName(MlirRegisteredOperationName name);

#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
