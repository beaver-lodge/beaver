#ifndef MLIR_C_BEAVER_INTERFACES_H
#define MLIR_C_BEAVER_INTERFACES_H

#include "mlir-c/IR.h"
#include "mlir-c/Interfaces.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

/// A callback-scoped collector owned by MLIR. It is valid only while a
/// MemoryEffectsOpInterface callback is running.
DEFINE_C_API_STRUCT(MlirBeaverMemoryEffectInstancesList, void);

#undef DEFINE_C_API_STRUCT

typedef enum {
  MlirBeaverMemoryEffectAllocate = 0,
  MlirBeaverMemoryEffectFree = 1,
  MlirBeaverMemoryEffectRead = 2,
  MlirBeaverMemoryEffectWrite = 3,
  MlirBeaverMemoryEffectUnknown = 4,
} MlirBeaverMemoryEffectKind;

typedef struct {
  void (*construct)(void *userData);
  void (*destruct)(void *userData);
  void (*getEffects)(MlirOperation op,
                     MlirBeaverMemoryEffectInstancesList effects,
                     void *userData);
  void *userData;
} MlirBeaverMemoryEffectsOpInterfaceCallbacks;

typedef void (*MlirBeaverMemoryEffectInstancesCallback)(
    intptr_t numEffects, MlirMemoryEffectInstance *effects, void *userData);

/// Stable Beaver-owned adapters for the MemoryEffects C API. Upstream changed
/// its collector ABI from an opaque list to a callback in August 2026; these
/// functions deliberately use MLIR's C++ interface so Beaver can compile
/// against both layouts without exposing either one to Zig or Elixir.
MLIR_CAPI_EXPORTED void beaverMemoryEffectsOpInterfaceAttachFallbackModel(
    MlirContext context, MlirStringRef operationName,
    MlirBeaverMemoryEffectsOpInterfaceCallbacks callbacks);

MLIR_CAPI_EXPORTED void beaverMemoryEffectInstancesListAppend(
    MlirBeaverMemoryEffectInstancesList effects,
    MlirMemoryEffectInstance instance);

MLIR_CAPI_EXPORTED bool beaverMemoryEffectsOpInterfaceGetEffects(
    MlirOperation operation, MlirBeaverMemoryEffectInstancesCallback callback,
    void *userData);

MLIR_CAPI_EXPORTED MlirBeaverMemoryEffectKind
beaverMemoryEffectInstanceGetKind(MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED MlirSideEffectResource
beaverMemoryEffectInstanceGetResource(MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED int
beaverMemoryEffectInstanceGetStage(MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED bool beaverMemoryEffectInstanceGetEffectOnFullRegion(
    MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED MlirAttribute
beaverMemoryEffectInstanceGetParameters(MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED MlirOpOperand
beaverMemoryEffectInstanceGetOpOperand(MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED MlirValue
beaverMemoryEffectInstanceGetValue(MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED MlirAttribute
beaverMemoryEffectInstanceGetSymbolRef(MlirMemoryEffectInstance instance);

MLIR_CAPI_EXPORTED void beaverTransformOnlyReadsHandle(
    MlirOpOperand *operands, intptr_t numOperands,
    MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformConsumesHandle(
    MlirOpOperand *operands, intptr_t numOperands,
    MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformProducesHandle(
    MlirValue *results, intptr_t numResults,
    MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformModifiesPayload(
    MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformOnlyReadsPayload(
    MlirBeaverMemoryEffectInstancesList effects);

/// Whether TileUsingForOp accepts packed tile-size and interchange handles.
MLIR_CAPI_EXPORTED bool beaverTransformPackedParamsSupported(void);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_INTERFACES_H
