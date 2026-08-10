#include "mlir-c/Beaver/LLVMIR.h"

#include "llvm-c/Core.h"
#include "llvm-c/IRReader.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "mlir-c/Target/LLVMIR.h"

#include <cstring>
#include <mutex>
#include <string>

namespace {

void initializeNVPTX() {
  static std::once_flag once;
  std::call_once(once, [] {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

void emitString(MlirStringCallback callback, void *userData,
                const char *message) {
  if (callback && message)
    callback(mlirStringRefCreate(message, std::strlen(message)), userData);
}

MlirLogicalResult failWithMessage(MlirStringCallback callback, void *userData,
                                  char *message) {
  emitString(callback, userData,
             message ? message : "LLVM target compilation failed");
  if (message)
    LLVMDisposeMessage(message);
  return mlirLogicalResultFailure();
}

} // namespace

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

MlirLogicalResult beaverCompileLLVMIRToPTX(
    MlirStringRef llvmIR, MlirStringRef cpu, MlirStringRef features,
    MlirStringCallback outputCallback, MlirStringCallback errorCallback,
    void *userData) {
  if (!llvmIR.data || !outputCallback)
    return mlirLogicalResultFailure();

  initializeNVPTX();

  const std::string triple = "nvptx64-nvidia-cuda";
  const std::string cpuString(cpu.data ? cpu.data : "", cpu.length);
  const std::string featureString(features.data ? features.data : "",
                                  features.length);

  LLVMContextRef context = LLVMContextCreate();
  LLVMMemoryBufferRef input = LLVMCreateMemoryBufferWithMemoryRangeCopy(
      llvmIR.data, llvmIR.length, "beaver.ll");
  LLVMModuleRef module = nullptr;
  char *message = nullptr;

  if (LLVMParseIRInContext2(context, input, &module, &message)) {
    LLVMDisposeMemoryBuffer(input);
    LLVMContextDispose(context);
    return failWithMessage(errorCallback, userData, message);
  }
  LLVMDisposeMemoryBuffer(input);

  LLVMTargetRef target = nullptr;
  if (LLVMGetTargetFromTriple(triple.c_str(), &target, &message)) {
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    return failWithMessage(errorCallback, userData, message);
  }

  LLVMTargetMachineRef machine = LLVMCreateTargetMachine(
      target, triple.c_str(), cpuString.c_str(), featureString.c_str(),
      LLVMCodeGenLevelDefault, LLVMRelocDefault, LLVMCodeModelDefault);
  if (!machine) {
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    emitString(errorCallback, userData, "failed to create NVPTX target machine");
    return mlirLogicalResultFailure();
  }

  LLVMSetTarget(module, triple.c_str());
  LLVMTargetDataRef layout = LLVMCreateTargetDataLayout(machine);
  LLVMSetModuleDataLayout(module, layout);
  LLVMDisposeTargetData(layout);

  LLVMMemoryBufferRef output = nullptr;
  if (LLVMTargetMachineEmitToMemoryBuffer(machine, module, LLVMAssemblyFile,
                                          &message, &output)) {
    LLVMDisposeTargetMachine(machine);
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    return failWithMessage(errorCallback, userData, message);
  }

  outputCallback(mlirStringRefCreate(LLVMGetBufferStart(output),
                                     LLVMGetBufferSize(output)),
                 userData);
  LLVMDisposeMemoryBuffer(output);
  LLVMDisposeTargetMachine(machine);
  LLVMDisposeModule(module);
  LLVMContextDispose(context);
  return mlirLogicalResultSuccess();
}
