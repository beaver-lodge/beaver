#ifndef MLIR_C_BEAVER_TRITON_H_
#define MLIR_C_BEAVER_TRITON_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register Triton's core dialects (tt, ttir, ttgir, ttng, ttinstrument,
/// gluon) plus the upstream dialects they depend on on `context`, and load
/// them. Requires a build linked against the Triton core prebuilt; otherwise
/// this returns false.
MLIR_CAPI_EXPORTED bool beaverContextRegisterTritonDialects(MlirContext context);

/// Register Triton's core passes and conversions in the global MLIR pass
/// registry so pass pipelines can refer to them by name. Idempotent per MLIR
/// registration semantics; call before running Triton pipelines. Returns false
/// when built without Triton support.
MLIR_CAPI_EXPORTED bool beaverRegisterTritonPasses();

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_TRITON_H_
