#ifndef MLIR_C_BEAVER_LLVM_IR_H
#define MLIR_C_BEAVER_LLVM_IR_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Translates an MLIR module to LLVM IR and streams its textual form to the
/// callback. All LLVM objects and the printed message are owned and released
/// by this call. Failure is reported without invoking the callback.
MLIR_CAPI_EXPORTED MlirLogicalResult beaverTranslateModuleToLLVMIRText(
    MlirOperation module, MlirStringCallback callback, void *userData);

/// Compiles textual LLVM IR to NVPTX assembly and streams the result to
/// outputCallback. Diagnostics are streamed to errorCallback. All temporary
/// LLVM objects and buffers are owned and released by this call.
MLIR_CAPI_EXPORTED MlirLogicalResult beaverCompileLLVMIRToPTX(
    MlirStringRef llvmIR, MlirStringRef cpu, MlirStringRef features,
    MlirStringCallback outputCallback, MlirStringCallback errorCallback,
    void *userData);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_LLVM_IR_H
