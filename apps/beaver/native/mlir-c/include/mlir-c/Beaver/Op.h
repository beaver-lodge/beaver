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
DEFINE_C_API_STRUCT(MlirOperand, void);

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

MLIR_CAPI_EXPORTED MlirOperand beaverValueGetFirstOperand(MlirValue value);

MLIR_CAPI_EXPORTED MlirOperand beaverOperandGetNext(MlirOperand operand);

MLIR_CAPI_EXPORTED bool beaverOperandIsNull(MlirOperand operand);

MLIR_CAPI_EXPORTED MlirValue beaverOperandGetValue(MlirOperand operand);

MLIR_CAPI_EXPORTED MlirOperation beaverOperandGetOwner(MlirOperand operand);

MLIR_CAPI_EXPORTED uint32_t beaverOperandGetNumber(MlirOperand operand);

#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
