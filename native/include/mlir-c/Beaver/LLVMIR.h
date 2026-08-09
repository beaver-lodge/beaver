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

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_LLVM_IR_H
