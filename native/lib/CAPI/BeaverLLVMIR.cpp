#include "mlir-c/Beaver/LLVMIR.h"

#include "llvm-c/Core.h"
#include "mlir-c/Target/LLVMIR.h"

#include <cstring>

MlirLogicalResult beaverTranslateModuleToLLVMIRText(
    MlirOperation module, MlirStringCallback callback, void *userData) {
  if (!module.ptr || !callback)
    return mlirLogicalResultFailure();

  LLVMContextRef llvmContext = LLVMContextCreate();
  if (!llvmContext)
    return mlirLogicalResultFailure();

  LLVMModuleRef llvmModule = mlirTranslateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    LLVMContextDispose(llvmContext);
    return mlirLogicalResultFailure();
  }

  char *message = LLVMPrintModuleToString(llvmModule);
  if (!message) {
    LLVMDisposeModule(llvmModule);
    LLVMContextDispose(llvmContext);
    return mlirLogicalResultFailure();
  }

  callback(mlirStringRefCreate(message, std::strlen(message)), userData);
  LLVMDisposeMessage(message);
  LLVMDisposeModule(llvmModule);
  LLVMContextDispose(llvmContext);
  return mlirLogicalResultSuccess();
}
