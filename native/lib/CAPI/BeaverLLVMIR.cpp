#include "mlir-c/Beaver/LLVMIR.h"

#include "llvm-c/Core.h"
#include "llvm-c/Error.h"
#include "llvm-c/IRReader.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "mlir-c/Target/LLVMIR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <chrono>
#include <cstring>
#include <mutex>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
uint64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
void initializeNVPTX() {
  static std::once_flag once;
  std::call_once(once, [] {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}
void initializeNativeTarget() {
  static std::once_flag once;
  std::call_once(once, [] {
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
  });
}
void emitString(MlirStringCallback callback, void *userData, const char *message) {
  if (callback && message) callback(mlirStringRefCreate(message, std::strlen(message)), userData);
}
MlirLogicalResult failWithMessage(MlirStringCallback callback, void *userData, char *message) {
  emitString(callback, userData, message ? message : "LLVM target compilation failed");
  if (message) LLVMDisposeMessage(message);
  return mlirLogicalResultFailure();
}
} // namespace
MlirLogicalResult beaverTranslateModuleToLLVMIRText(MlirOperation module, MlirStringCallback callback, void *userData) {
  if (!module.ptr || !callback) return mlirLogicalResultFailure();
  LLVMContextRef llvmContext = LLVMContextCreate();
  if (!llvmContext) return mlirLogicalResultFailure();
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
MlirLogicalResult beaverCompileLLVMIRToPTX(MlirStringRef llvmIR, MlirStringRef cpu, MlirStringRef features,
                                           MlirStringCallback outputCallback, MlirStringCallback errorCallback,
                                           void *userData) {
  if (!llvmIR.data || !outputCallback) return mlirLogicalResultFailure();
  initializeNVPTX();
  const std::string triple = "nvptx64-nvidia-cuda";
  const std::string cpuString(cpu.data ? cpu.data : "", cpu.length);
  const std::string featureString(features.data ? features.data : "", features.length);
  LLVMContextRef context = LLVMContextCreate();
  LLVMMemoryBufferRef input = LLVMCreateMemoryBufferWithMemoryRangeCopy(llvmIR.data, llvmIR.length, "beaver.ll");
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
  LLVMTargetMachineRef machine =
      LLVMCreateTargetMachine(target, triple.c_str(), cpuString.c_str(), featureString.c_str(), LLVMCodeGenLevelDefault,
                              LLVMRelocDefault, LLVMCodeModelDefault);
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
  if (LLVMTargetMachineEmitToMemoryBuffer(machine, module, LLVMAssemblyFile, &message, &output)) {
    LLVMDisposeTargetMachine(machine);
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    return failWithMessage(errorCallback, userData, message);
  }
  outputCallback(mlirStringRefCreate(LLVMGetBufferStart(output), LLVMGetBufferSize(output)), userData);
  LLVMDisposeMemoryBuffer(output);
  LLVMDisposeTargetMachine(machine);
  LLVMDisposeModule(module);
  LLVMContextDispose(context);
  return mlirLogicalResultSuccess();
}
extern "C" MLIR_CAPI_EXPORTED void beaverLLVMGetHostTarget(MlirStringCallback callback, void *userData) {
  if (!callback) return;
  initializeNativeTarget();
  char *triple = LLVMGetDefaultTargetTriple();
  char *cpu = LLVMGetHostCPUName();
  char *features = LLVMGetHostCPUFeatures();
  emitString(callback, userData, triple ? triple : "");
  emitString(callback, userData, cpu ? cpu : "");
  emitString(callback, userData, features ? features : "");
  if (triple) LLVMDisposeMessage(triple);
  if (cpu) LLVMDisposeMessage(cpu);
  if (features) LLVMDisposeMessage(features);
}
extern "C" MLIR_CAPI_EXPORTED MlirLogicalResult beaverCompileToObject(
    MlirOperation moduleOp, MlirStringRef llvmIR, MlirStringRef tripleRef,
    MlirStringRef cpuRef, MlirStringRef featuresRef, unsigned optLevel, bool pic, bool emitAsm, uint64_t *timingsNs,
    MlirStringCallback outputCallback, MlirStringCallback errorCallback, void *userData) {
  if (!outputCallback) return mlirLogicalResultFailure();
  if (timingsNs) {
    timingsNs[0] = 0;
    timingsNs[1] = 0;
    timingsNs[2] = 0;
  }
  initializeNativeTarget();
  initializeNVPTX();
  char *defaultTriple = LLVMGetDefaultTargetTriple();
  char *hostCpu = LLVMGetHostCPUName();
  char *hostFeatures = LLVMGetHostCPUFeatures();
  std::string triple = tripleRef.length > 0 ? std::string(tripleRef.data, tripleRef.length)
                                            : std::string(defaultTriple ? defaultTriple : "");
  bool isHostTriple = defaultTriple && (triple == defaultTriple);
  std::string cpu = cpuRef.length > 0 ? std::string(cpuRef.data, cpuRef.length)
                                      : (isHostTriple && hostCpu ? std::string(hostCpu) : "");
  std::string features = featuresRef.length > 0 ? std::string(featuresRef.data, featuresRef.length)
                                                : (isHostTriple && hostFeatures ? std::string(hostFeatures) : "");
  if (defaultTriple) LLVMDisposeMessage(defaultTriple);
  if (hostCpu) LLVMDisposeMessage(hostCpu);
  if (hostFeatures) LLVMDisposeMessage(hostFeatures);
  uint64_t t0 = nowNs();
  LLVMContextRef context = LLVMContextCreate();
  LLVMModuleRef module = nullptr;
  char *message = nullptr;
  if (moduleOp.ptr) {
    module = mlirTranslateModuleToLLVMIR(moduleOp, context);
    if (!module) {
      LLVMContextDispose(context);
      emitString(errorCallback, userData, "failed to translate MLIR module to LLVM IR");
      return mlirLogicalResultFailure();
    }
  } else if (llvmIR.data) {
    LLVMMemoryBufferRef input = LLVMCreateMemoryBufferWithMemoryRangeCopy(llvmIR.data, llvmIR.length, "beaver_aot.ll");
    if (LLVMParseIRInContext2(context, input, &module, &message)) {
      LLVMDisposeMemoryBuffer(input);
      LLVMContextDispose(context);
      return failWithMessage(errorCallback, userData, message);
    }
    LLVMDisposeMemoryBuffer(input);
  } else {
    LLVMContextDispose(context);
    emitString(errorCallback, userData, "neither MLIR module nor LLVM IR string provided");
    return mlirLogicalResultFailure();
  }
  uint64_t t1 = nowNs();
  if (timingsNs) timingsNs[0] = t1 - t0;
  LLVMTargetRef target = nullptr;
  if (LLVMGetTargetFromTriple(triple.c_str(), &target, &message)) {
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    return failWithMessage(errorCallback, userData, message);
  }
  LLVMCodeGenOptLevel cgOpt = optLevel == 0   ? LLVMCodeGenLevelNone
                              : optLevel == 1 ? LLVMCodeGenLevelLess
                              : optLevel == 2 ? LLVMCodeGenLevelDefault
                                              : LLVMCodeGenLevelAggressive;
  LLVMRelocMode reloc = pic ? LLVMRelocPIC : LLVMRelocDefault;
  LLVMTargetMachineRef machine = LLVMCreateTargetMachine(
      target, triple.c_str(), cpu.c_str(), features.c_str(), cgOpt, reloc, LLVMCodeModelDefault);
  if (!machine) {
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    emitString(errorCallback, userData, "failed to create LLVM target machine");
    return mlirLogicalResultFailure();
  }
  LLVMSetTarget(module, triple.c_str());
  LLVMTargetDataRef layout = LLVMCreateTargetDataLayout(machine);
  LLVMSetModuleDataLayout(module, layout);
  LLVMDisposeTargetData(layout);
  uint64_t tOptStart = nowNs();
  if (optLevel > 0) {
    const char *passes = optLevel == 1   ? "default<O1>"
                         : optLevel == 2 ? "default<O2>"
                                         : "default<O3>";
    LLVMPassBuilderOptionsRef pbOpts = LLVMCreatePassBuilderOptions();
    if (LLVMErrorRef err = LLVMRunPasses(module, passes, machine, pbOpts)) {
      char *errMsg = LLVMGetErrorMessage(err);
      LLVMDisposePassBuilderOptions(pbOpts);
      LLVMDisposeTargetMachine(machine);
      LLVMDisposeModule(module);
      LLVMContextDispose(context);
      return failWithMessage(errorCallback, userData, errMsg);
    }
    LLVMDisposePassBuilderOptions(pbOpts);
  }
  uint64_t tOptEnd = nowNs();
  if (timingsNs) timingsNs[1] = tOptEnd - tOptStart;
  LLVMMemoryBufferRef output = nullptr;
  LLVMCodeGenFileType fileType = emitAsm ? LLVMAssemblyFile : LLVMObjectFile;
  if (LLVMTargetMachineEmitToMemoryBuffer(machine, module, fileType, &message, &output)) {
    LLVMDisposeTargetMachine(machine);
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
    return failWithMessage(errorCallback, userData, message);
  }
  uint64_t tCodegenEnd = nowNs();
  if (timingsNs) timingsNs[2] = tCodegenEnd - tOptEnd;
  outputCallback(mlirStringRefCreate(LLVMGetBufferStart(output), LLVMGetBufferSize(output)), userData);
  LLVMDisposeMemoryBuffer(output);
  LLVMDisposeTargetMachine(machine);
  LLVMDisposeModule(module);
  LLVMContextDispose(context);
  return mlirLogicalResultSuccess();
}
extern "C" MLIR_CAPI_EXPORTED MlirModule beaverExtractFunctionModule(
    MlirOperation moduleOp, MlirStringRef funcNameRef, bool keepGlobals) {
  if (!moduleOp.ptr) return {nullptr};
  auto mod = llvm::dyn_cast<mlir::ModuleOp>(unwrap(moduleOp));
  if (!mod) return {nullptr};
  llvm::StringRef targetName = unwrap(funcNameRef);
  mlir::ModuleOp cloned = llvm::cast<mlir::ModuleOp>(mod.getOperation()->clone());
  bool found = false;
  for (mlir::Operation &op : llvm::make_early_inc_range(cloned.getBody()->getOperations())) {
    if (auto llvmFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(&op)) {
      if (llvmFunc.getSymName() == targetName) {
        if (!llvmFunc.isExternal()) {
          found = true;
          if (llvmFunc.getLinkage() == mlir::LLVM::Linkage::Internal ||
              llvmFunc.getLinkage() == mlir::LLVM::Linkage::Private) {
            llvmFunc.setLinkage(mlir::LLVM::Linkage::External);
          }
        }
      } else {
        if (!llvmFunc.isExternal()) {
          llvmFunc.getBody().dropAllReferences();
          llvmFunc.getBody().getBlocks().clear();
          llvmFunc.setLinkage(mlir::LLVM::Linkage::External);
        }
      }
    } else if (auto fnOp = llvm::dyn_cast<mlir::FunctionOpInterface>(&op)) {
      if (fnOp.getName() == targetName) { if (!fnOp.isExternal()) found = true; } else if (!fnOp.isExternal()) {
        fnOp.getFunctionBody().dropAllReferences();
        fnOp.getFunctionBody().getBlocks().clear();
        mlir::SymbolTable::setSymbolVisibility(&op, mlir::SymbolTable::Visibility::Private);
      }
    } else if (!keepGlobals) {
      if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(&op)) {
        if (globalOp.getLinkage() == mlir::LLVM::Linkage::Internal ||
            globalOp.getLinkage() == mlir::LLVM::Linkage::Private) { continue; }
        globalOp.removeValueAttr();
        if (!globalOp.getInitializerRegion().empty()) {
          globalOp.getInitializerRegion().dropAllReferences();
          globalOp.getInitializerRegion().getBlocks().clear();
        }
        globalOp.setLinkage(mlir::LLVM::Linkage::External);
      }
    }
  }
  if (!found) {
    cloned.erase();
    return {nullptr};
  }
  return {cloned.getOperation()};
}
