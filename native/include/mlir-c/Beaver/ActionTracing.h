#ifndef MLIR_C_BEAVER_ACTIONTRACING_H_
#define MLIR_C_BEAVER_ACTIONTRACING_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a context-scoped action tracing session.
struct MlirBeaverActionTracing {
  void *ptr;
};
typedef struct MlirBeaverActionTracing MlirBeaverActionTracing;

/// Callback receiving the serialized JSON array of drained action events.
/// `data` points to a NUL-terminated string valid only for the duration of the
/// call.
typedef void (*MlirBeaverActionEventsCallback)(const char *data, void *user_data);

/// Attach an action tracing session to `context`.
///
/// `filter_json` is a JSON array of action tags to observe; empty string
/// observes all actions. `location_json` is a JSON array of source location
/// substrings; an action is observed only when one of its context IR units has
/// a matching location. `skip_json` is a JSON object mapping an action tag to a
/// non-negative skip count: the first N occurrences of that tag are dropped
/// (not executed). `limit_json` is a JSON object mapping an action tag to a
/// non-negative execution limit; once reached, further occurrences of that tag
/// are skipped.
///
/// The returned handle owns the tracing session; it must be released with
/// mlirBeaverActionTracingDetach. Only one tracing session may be attached to
/// a context at a time.
MLIR_CAPI_EXPORTED MlirBeaverActionTracing
mlirBeaverActionTracingAttach(MlirContext context, MlirStringRef filter_json,
                              MlirStringRef location_json,
                              MlirStringRef skip_json, MlirStringRef limit_json);

/// Drain all pending action events from the session and hand them to
/// `callback` as a JSON array. Returns true on success.
MLIR_CAPI_EXPORTED bool mlirBeaverActionTracingDrain(
    MlirBeaverActionTracing tracing, MlirBeaverActionEventsCallback callback,
    void *user_data);

/// Detach and destroy the tracing session. Safe to call from any thread.
MLIR_CAPI_EXPORTED void
mlirBeaverActionTracingDetach(MlirBeaverActionTracing tracing);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_ACTIONTRACING_H_
