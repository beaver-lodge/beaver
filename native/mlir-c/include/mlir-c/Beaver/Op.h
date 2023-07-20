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

MLIR_CAPI_EXPORTED void
beaverRegisteredOperationsOfDialect(MlirContext context, MlirStringRef dialect,
                                    MlirRegisteredOperationName *ret,
                                    size_t *num);

MLIR_CAPI_EXPORTED void
beaverRegisteredDialects(MlirContext context, MlirStringRef *ret, size_t *num);
MLIR_CAPI_EXPORTED MlirOperand beaverValueGetFirstOperand(MlirValue value);

MLIR_CAPI_EXPORTED MlirOperand beaverOperandGetNext(MlirOperand operand);

MLIR_CAPI_EXPORTED bool beaverOperandIsNull(MlirOperand operand);

MLIR_CAPI_EXPORTED MlirValue beaverOperandGetValue(MlirOperand operand);

MLIR_CAPI_EXPORTED MlirOperation beaverOperandGetOwner(MlirOperand operand);

MLIR_CAPI_EXPORTED intptr_t beaverOperandGetNumber(MlirOperand operand);

MLIR_CAPI_EXPORTED const char *beaverStringRefGetData(MlirStringRef string_ref);
MLIR_CAPI_EXPORTED size_t beaverStringRefGetLength(MlirStringRef string_ref);

MLIR_CAPI_EXPORTED bool beaverContextIsNull(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverDialectIsNull(MlirDialect dialect);
MLIR_CAPI_EXPORTED bool
beaverDialectRegistryIsNull(MlirDialectRegistry registry);
MLIR_CAPI_EXPORTED bool beaverLocationIsNull(MlirLocation location);
MLIR_CAPI_EXPORTED bool beaverModuleIsNull(MlirModule module);
MLIR_CAPI_EXPORTED bool beaverOperationIsNull(MlirOperation op);
MLIR_CAPI_EXPORTED bool beaverRegionIsNull(MlirRegion region);
MLIR_CAPI_EXPORTED bool beaverBlockIsNull(MlirBlock block);
MLIR_CAPI_EXPORTED bool beaverValueIsNull(MlirValue value);
MLIR_CAPI_EXPORTED bool beaverTypeIsNull(MlirType type);
MLIR_CAPI_EXPORTED bool beaverAttributeIsNull(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool beaverSymbolTableIsNull(MlirSymbolTable symbolTable);

MLIR_CAPI_EXPORTED MlirStringRef
beaverMlirOperationStateGetName(MlirOperationState state);
MLIR_CAPI_EXPORTED MlirContext
beaverMlirOperationStateGetContext(MlirOperationState state);
MLIR_CAPI_EXPORTED MlirLocation
beaverMlirOperationStateGetLocation(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumResults(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumOperands(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumRegions(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumAttributes(MlirOperationState state);

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsSuccess(MlirLogicalResult res);
MLIR_CAPI_EXPORTED bool beaverLogicalResultIsFailure(MlirLogicalResult res);
MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultSuccess();
MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultFailure();

MLIR_CAPI_EXPORTED MlirIdentifier beaverOperationGetName(MlirOperation op,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED MlirAttribute beaverOperationGetAttribute(MlirOperation op,
                                                             intptr_t pos);

MLIR_CAPI_EXPORTED
MlirIdentifier beaverMlirNamedAttributeGetName(MlirNamedAttribute);
MLIR_CAPI_EXPORTED

MLIR_CAPI_EXPORTED
MlirAttribute beaverMlirNamedAttributeGetAttribute(MlirNamedAttribute na);

MLIR_CAPI_EXPORTED MlirPass beaverCreateExternalPass(
    void (*construct)(void *userData), MlirTypeID passID, MlirStringRef name,
    MlirStringRef argument, MlirStringRef description, MlirStringRef opName,
    intptr_t nDependentDialects, MlirDialectHandle *dependentDialects,
    void (*destruct)(void *userData),
    MlirLogicalResult (*initialize)(MlirContext ctx, void *userData),
    void *(*clone)(void *userData),
    void (*run)(MlirOperation op, MlirExternalPass pass, void *userData),
    void *userData);

MLIR_CAPI_EXPORTED MlirAttribute beaverGetReassociationIndicesForReshape(
    MlirType sourceType, MlirType targetType);

MLIR_CAPI_EXPORTED void beaverLocationPrint(MlirLocation location,
                                            MlirStringCallback callback,
                                            void *userData);

MLIR_CAPI_EXPORTED void beaverOperationDumpGeneric(MlirOperation op);

#include "mlir-c/ExecutionEngine.h"

MLIR_CAPI_EXPORTED bool beaverMlirExecutionEngineIsNull(MlirExecutionEngine w);
#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
