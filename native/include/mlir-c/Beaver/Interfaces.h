#ifndef MLIR_C_BEAVER_INTERFACES_H
#define MLIR_C_BEAVER_INTERFACES_H

#include "mlir-c/IR.h"
#include "mlir-c/Interfaces.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                                                             \
  struct name {                                                                                                        \
    storage *ptr;                                                                                                      \
  };                                                                                                                   \
  typedef struct name name

/// A callback-scoped collector forwarding to MLIR. It is valid only while a
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

// The BEAM collector forwards effects to MLIR's callback, which copies them.
typedef struct {
  MlirMemoryEffectInstancesCallback callback;
  void *userData;
} MlirBeaverMemoryEffectSink;

MLIR_CAPI_EXPORTED void beaverMemoryEffectInstancesListAppend(MlirBeaverMemoryEffectInstancesList effects,
                                                              MlirMemoryEffectInstance instance);

MLIR_CAPI_EXPORTED MlirBeaverMemoryEffectKind beaverMemoryEffectInstanceGetKind(MlirMemoryEffectInstance instance);
MLIR_CAPI_EXPORTED MlirOpOperand beaverMemoryEffectInstanceGetOpOperand(MlirMemoryEffectInstance instance);

MLIR_CAPI_EXPORTED void beaverTransformOnlyReadsHandle(MlirOpOperand *operands, intptr_t numOperands,
                                                       MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformConsumesHandle(MlirOpOperand *operands, intptr_t numOperands,
                                                      MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformProducesHandle(MlirValue *results, intptr_t numResults,
                                                      MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformModifiesPayload(MlirBeaverMemoryEffectInstancesList effects);
MLIR_CAPI_EXPORTED void beaverTransformOnlyReadsPayload(MlirBeaverMemoryEffectInstancesList effects);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_INTERFACES_H
