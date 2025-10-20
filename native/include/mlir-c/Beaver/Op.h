#ifndef APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
#define APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_

#include "mlir-c/Pass.h"
#include "mlir-c/Rewrite.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED bool beaverIsOpNameTerminator(MlirStringRef op_name,
                                                 MlirContext context);

MLIR_CAPI_EXPORTED void beaverContextGetOps(MlirContext context,
                                            MlirStringCallback insert,
                                            void *container);

MLIR_CAPI_EXPORTED void beaverContextGetDialects(MlirContext context,
                                                 MlirStringCallback insert,
                                                 void *container);

MLIR_CAPI_EXPORTED const char *beaverStringRefGetData(MlirStringRef string_ref);
MLIR_CAPI_EXPORTED size_t beaverStringRefGetLength(MlirStringRef string_ref);

MLIR_CAPI_EXPORTED bool beaverIsNullContext(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverIsNullDialect(MlirDialect dialect);
MLIR_CAPI_EXPORTED bool
beaverIsNullDialectRegistry(MlirDialectRegistry registry);
MLIR_CAPI_EXPORTED bool beaverIsNullLocation(MlirLocation location);
MLIR_CAPI_EXPORTED bool beaverIsNullModule(MlirModule module);
MLIR_CAPI_EXPORTED bool beaverIsNullOperation(MlirOperation op);
MLIR_CAPI_EXPORTED bool beaverIsNullRegion(MlirRegion region);
MLIR_CAPI_EXPORTED bool beaverIsNullBlock(MlirBlock block);
MLIR_CAPI_EXPORTED bool beaverIsNullValue(MlirValue value);
MLIR_CAPI_EXPORTED bool beaverIsNullType(MlirType type);
MLIR_CAPI_EXPORTED bool beaverIsNullAttribute(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool beaverIsNullSymbolTable(MlirSymbolTable symbolTable);

MLIR_CAPI_EXPORTED MlirStringRef
beaverOperationStateGetName(MlirOperationState state);
MLIR_CAPI_EXPORTED MlirContext
beaverOperationStateGetContext(MlirOperationState state);
MLIR_CAPI_EXPORTED MlirLocation
beaverOperationStateGetLocation(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumResults(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumOperands(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumRegions(MlirOperationState state);
MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumAttributes(MlirOperationState state);

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsSuccess(MlirLogicalResult res);
MLIR_CAPI_EXPORTED bool beaverLogicalResultIsFailure(MlirLogicalResult res);

MLIR_CAPI_EXPORTED
MlirIdentifier beaverNamedAttributeGetName(MlirNamedAttribute na);
MLIR_CAPI_EXPORTED

MLIR_CAPI_EXPORTED
MlirAttribute beaverNamedAttributeGetAttribute(MlirNamedAttribute na);

MLIR_CAPI_EXPORTED MlirPass beaverPassCreate(
    void (*construct)(void *userData), void (*destruct)(void *userData),
    MlirLogicalResult (*initialize)(MlirContext ctx, void *userData),
    void *(*clone)(void *userData),
    void (*run)(MlirOperation op, MlirExternalPass pass, void *userData),
    MlirTypeID passID, MlirStringRef name, MlirStringRef argument,
    MlirStringRef description, MlirStringRef opName,
    intptr_t nDependentDialects, MlirDialectHandle *dependentDialects,
    void *userData);

MLIR_CAPI_EXPORTED MlirAttribute beaverGetReassociationIndicesForReshape(
    MlirType sourceType, MlirType targetType);

MLIR_CAPI_EXPORTED void beaverLocationPrint(MlirLocation location,
                                            MlirStringCallback callback,
                                            void *userData);
MLIR_CAPI_EXPORTED void mlirIdentifierPrint(MlirIdentifier identifier,
                                            MlirStringCallback callback,
                                            void *userData);
MLIR_CAPI_EXPORTED void beaverOperationPrintSpecializedFrom(
    MlirOperation op, MlirStringCallback callback, void *userData);
MLIR_CAPI_EXPORTED void
beaverOperationPrintGenericOpForm(MlirOperation op, MlirStringCallback callback,
                                  void *userData);
MLIR_CAPI_EXPORTED void beaverOperationDumpGeneric(MlirOperation op);
MLIR_CAPI_EXPORTED MlirType beaverIRDLGetDefinedType(MlirStringRef dialect,
                                                     MlirStringRef type,
                                                     MlirAttribute params);
MLIR_CAPI_EXPORTED MlirAttribute beaverIRDLGetDefinedAttr(MlirStringRef dialect,
                                                          MlirStringRef attr,
                                                          MlirAttribute params);

MLIR_CAPI_EXPORTED MlirGreedyRewriteDriverConfig
beaverGreedyRewriteDriverConfigGet();
MLIR_CAPI_EXPORTED MlirType beaverDenseElementsAttrGetType(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t beaverShapedTypeGetNumElements(MlirType type);

#include "mlir-c/ExecutionEngine.h"

MLIR_CAPI_EXPORTED bool beaverIsNullExecutionEngine(MlirExecutionEngine w);
#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
