#ifndef APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
#define APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_

#include <stdint.h>

#include "mlir-c/IR.h"
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

MLIR_CAPI_EXPORTED void beaverContextGetOps(MlirContext context,
                                            MlirStringCallback insert,
                                            void *container);

MLIR_CAPI_EXPORTED void beaverContextGetDialects(MlirContext context,
                                                 MlirStringCallback insert,
                                                 void *container);

MLIR_CAPI_EXPORTED const char *beaverStringRefGetData(MlirStringRef string_ref);
MLIR_CAPI_EXPORTED size_t beaverStringRefGetLength(MlirStringRef string_ref);

// Normalize MLIR's platform-sized structural hash to a width that can be
// represented safely by the BEAM NIF codec on both 32-bit and 64-bit hosts.
MLIR_CAPI_EXPORTED uint64_t
beaverOperationStructuralHashValue(MlirOperation op, uint32_t flags);

// MLIR's IRMapping::clear currently clears only value mappings despite the C
// API documenting all mappings. Keep Beaver's high-level clear contract whole.
MLIR_CAPI_EXPORTED void beaverIRMappingClear(MlirIRMapping mapping);

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

typedef enum {
  MlirBeaverSymbolVisibilityPublic = 0,
  MlirBeaverSymbolVisibilityPrivate = 1,
  MlirBeaverSymbolVisibilityNested = 2,
} MlirBeaverSymbolVisibility;

/// Returns the default attribute name used by SymbolOpInterface's default
/// visibility implementation. Custom symbol operations may store visibility
/// elsewhere; use the get/set functions below to inspect their visibility.
MLIR_CAPI_EXPORTED MlirStringRef
beaverSymbolTableGetDefaultVisibilityAttributeName(void);
MLIR_CAPI_EXPORTED MlirBeaverSymbolVisibility
beaverSymbolTableGetSymbolVisibility(MlirOperation symbol);
MLIR_CAPI_EXPORTED void beaverSymbolTableSetSymbolVisibility(
    MlirOperation symbol, MlirBeaverSymbolVisibility visibility);

MLIR_CAPI_EXPORTED MlirStringRef
beaverOperationStateGetName(MlirOperationState state);
// Create from a by-value state so the BEAM resource backing the state remains
// a live NIF argument for the entire operation creation call.
MLIR_CAPI_EXPORTED MlirOperation
beaverOperationCreate(MlirOperationState state);
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

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultSuccess(void);
MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultFailure(void);
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
// MLIR's upstream C API exposes fused children through a caller-owned output
// buffer. An indexed accessor keeps that buffer and its lifetime behind
// Beaver's native ABI boundary.
MLIR_CAPI_EXPORTED MlirLocation
beaverLocationFusedGetLocationAt(MlirLocation location, intptr_t position);
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

MLIR_CAPI_EXPORTED MlirStringRef beaverGetNumWorkgroupAttributionsAttrName();
MLIR_CAPI_EXPORTED MlirStringRef beaverGetContainerModuleAttrName();
MLIR_CAPI_EXPORTED MlirAttribute
beaverGPUObjectAttrGet(MlirAttribute target, int32_t format,
                       MlirStringRef object);
MLIR_CAPI_EXPORTED bool beaverAttributeIsAGPUObject(MlirAttribute attribute);
MLIR_CAPI_EXPORTED MlirStringRef
beaverGPUObjectAttrGetObject(MlirAttribute attribute);

#include "mlir-c/ExecutionEngine.h"

MLIR_CAPI_EXPORTED bool beaverIsNullExecutionEngine(MlirExecutionEngine w);
#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_OP_H_
