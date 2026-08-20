#ifndef APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_
#define APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Schedules callback-bridging work on the context's LLVM pool, outside BEAM
/// scheduler threads. An elastic pool prevents nested pass/rewrite callbacks
/// from starving a pool shared by multiple contexts.
MLIR_CAPI_EXPORTED bool beaverContextAddWork(MlirContext context,
                                             void (*task)(void *), void *arg);

/// Stable, capability-gated adapters for LLVM's transient MLIRContext scope.
/// Begin returns false when unsupported or when the context is already in a
/// scope; end returns false when no scope was entered through this adapter.
MLIR_CAPI_EXPORTED bool beaverContextTransientScopeSupported(void);
MLIR_CAPI_EXPORTED bool
beaverContextBeginTransientScope(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverContextEndTransientScope(MlirContext context);
MLIR_CAPI_EXPORTED bool
beaverContextHasActiveTransientScope(MlirContext context);

/// Creates a reusable thread pool that grows beyond its reported parallelism
/// when every worker is blocked by nested synchronous callback work.
MLIR_CAPI_EXPORTED MlirLlvmThreadPool
beaverLlvmThreadPoolCreateElastic(unsigned maxConcurrency);

/// Returns the LLVM version and source revision used to build Beaver. The
/// returned string has static storage duration and must not be freed.
MLIR_CAPI_EXPORTED MlirStringRef beaverGetLLVMVersion(void);

#ifdef __cplusplus
}
#endif

#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_
