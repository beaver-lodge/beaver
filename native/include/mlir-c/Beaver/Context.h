#ifndef APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_
#define APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_
#include "mlir-c/IR.h"
#ifdef __cplusplus
extern "C" {
#endif
/// Schedules work on Beaver's callback executor, outside BEAM scheduler threads. Requires thread-safe context locking,
/// but not MLIR internal parallelism. The caller must keep the context and borrowed IR alive until completion.
MLIR_CAPI_EXPORTED bool beaverContextAddWork(MlirContext context, void (*task)(void *), void *arg);
MLIR_CAPI_EXPORTED bool beaverExecutorConfigure(MlirContext context, unsigned capacity, unsigned idleMs);
MLIR_CAPI_EXPORTED bool beaverExecutorReleaseContext(MlirContext context);
MLIR_CAPI_EXPORTED void beaverExecutorSnapshot(MlirContext context, uint64_t *out);
MLIR_CAPI_EXPORTED void beaverExecutionMarkCallbacks(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverExecutionHasCallbacks(MlirContext context);
MLIR_CAPI_EXPORTED void *beaverExecutionMeasureBegin(void);
MLIR_CAPI_EXPORTED void beaverExecutionMeasureEnd(void *measurement, uint64_t *out);
MLIR_CAPI_EXPORTED uint64_t beaverExecutionCallbackBegin(void);
MLIR_CAPI_EXPORTED void beaverExecutionCallbackEnd(uint64_t start);
MLIR_CAPI_EXPORTED uint64_t beaverExecutionClock(void);
MLIR_CAPI_EXPORTED void *beaverExecutionBindOperation(MlirOperation operation);
MLIR_CAPI_EXPORTED void beaverExecutionUnbindOperation(void *binding);
MLIR_CAPI_EXPORTED void *beaverExecutionEnterOperation(MlirOperation operation);
MLIR_CAPI_EXPORTED void beaverExecutionLeaveOperation(void *scope);
MLIR_CAPI_EXPORTED MlirModule beaverModuleCreateParseTracked(MlirContext context, MlirStringRef source);
MLIR_CAPI_EXPORTED bool beaverOperationVerify(MlirOperation operation, bool recursive);
MLIR_CAPI_EXPORTED bool beaverContextEnableThreadSafeLocking(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverContextIsThreadSafeLockingEnabled(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverContextIsMultithreadingEnabled(MlirContext context);
/// Begin returns false when a transient scope is already active; end returns false when no scope was entered through
/// this adapter.
MLIR_CAPI_EXPORTED bool beaverContextBeginTransientScope(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverContextEndTransientScope(MlirContext context);
MLIR_CAPI_EXPORTED bool beaverContextHasActiveTransientScope(MlirContext context);
/// Creates a reusable thread pool that grows beyond its reported parallelism when every worker is blocked by nested
/// synchronous callback work.
MLIR_CAPI_EXPORTED MlirLlvmThreadPool beaverLlvmThreadPoolCreateElastic(unsigned maxConcurrency);
/// Returns the LLVM version and source revision used to build Beaver. The returned string has static storage duration
/// and must not be freed.
MLIR_CAPI_EXPORTED MlirStringRef beaverGetLLVMVersion(void);
#ifdef __cplusplus
}
#endif
#endif // APPS_BEAVER_CAPI_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_CONTEXT_H_
