#include "mlir-c/Beaver/LLVMIR.h"
#include "llvm-c/Core.h"
#include "llvm-c/Error.h"
#include "llvm-c/IRReader.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir-c/Target/LLVMIR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
namespace {
uint64_t nowNs() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
void initializeTargets() {
  static std::once_flag once;
  std::call_once(once, [] {
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}
void emitString(MlirStringCallback callback, void *data, const char *message) {
  if (callback && message) callback(mlirStringRefCreate(message, std::strlen(message)), data);
}
std::array<std::string, 3> hostTarget() {
  initializeTargets();
  std::array<std::string, 3> result;
  for (auto [index, value] : llvm::enumerate(
           std::array{LLVMGetDefaultTargetTriple(), LLVMGetHostCPUName(), LLVMGetHostCPUFeatures()})) {
    result[index] = value ? value : "";
    LLVMDisposeMessage(value);
  }
  return result;
}
// Both translation and target emission own the same LLVM lifetime. Destruction
// order keeps the context alive through every success and diagnostic path.
struct Compilation {
  LLVMContextRef context = LLVMContextCreate();
  LLVMModuleRef module = nullptr;
  LLVMTargetMachineRef machine = nullptr;
  ~Compilation() {
    if (machine) LLVMDisposeTargetMachine(machine);
    if (module) LLVMDisposeModule(module);
    LLVMContextDispose(context);
  }
};
MlirLogicalResult compile(MlirOperation moduleOp, MlirStringRef ir,
    const std::array<std::string, 3> &targetSpec, unsigned optLevel, LLVMCodeGenOptLevel cgOpt,
    bool pic, bool assembly, uint64_t *timings, MlirStringCallback output, MlirStringCallback error, void *data) {
  if (!output) return mlirLogicalResultFailure();
  if (timings) std::fill_n(timings, 3, 0);
  Compilation c;
  char *message = nullptr;
  auto cleanupMessage = llvm::scope_exit([&] { LLVMDisposeMessage(message); });
  auto fail = [&](const char *reason) {
    emitString(error, data, reason);
    return mlirLogicalResultFailure();
  };
  auto start = nowNs();
  if (moduleOp.ptr) {
    c.module = mlirTranslateModuleToLLVMIR(moduleOp, c.context);
    if (!c.module) return fail("failed to translate MLIR module to LLVM IR");
  } else if (ir.data) {
    auto input = LLVMCreateMemoryBufferWithMemoryRangeCopy(ir.data, ir.length, "beaver.ll");
    auto cleanupInput = llvm::scope_exit([&] { LLVMDisposeMemoryBuffer(input); });
    if (LLVMParseIRInContext2(c.context, input, &c.module, &message)) return fail(message);
  } else {
    return fail("neither MLIR module nor LLVM IR string provided");
  }
  if (timings) timings[0] = nowNs() - start;
  const auto &[triple, cpu, features] = targetSpec;
  LLVMTargetRef target = nullptr;
  if (LLVMGetTargetFromTriple(triple.c_str(), &target, &message)) return fail(message);
  c.machine = LLVMCreateTargetMachine(target, triple.c_str(), cpu.c_str(), features.c_str(), cgOpt,
                                     pic ? LLVMRelocPIC : LLVMRelocDefault, LLVMCodeModelDefault);
  if (!c.machine) return fail("failed to create LLVM target machine");
  LLVMSetTarget(c.module, triple.c_str());
  auto layout = LLVMCreateTargetDataLayout(c.machine);
  LLVMSetModuleDataLayout(c.module, layout);
  LLVMDisposeTargetData(layout);
  start = nowNs();
  if (optLevel > 0) {
    auto options = LLVMCreatePassBuilderOptions();
    auto cleanupOptions = llvm::scope_exit([&] { LLVMDisposePassBuilderOptions(options); });
    const char *passes = optLevel == 1 ? "default<O1>" : optLevel == 2 ? "default<O2>" : "default<O3>";
    if (auto err = LLVMRunPasses(c.module, passes, c.machine, options)) {
      char *reason = LLVMGetErrorMessage(err);
      auto result = fail(reason);
      LLVMDisposeErrorMessage(reason);
      return result;
    }
  }
  auto codegenStart = nowNs();
  if (timings) timings[1] = codegenStart - start;
  LLVMMemoryBufferRef buffer = nullptr;
  auto cleanupBuffer = llvm::scope_exit([&] { if (buffer) LLVMDisposeMemoryBuffer(buffer); });
  if (LLVMTargetMachineEmitToMemoryBuffer(c.machine, c.module, assembly ? LLVMAssemblyFile : LLVMObjectFile,
                                         &message, &buffer)) return fail(message);
  if (timings) timings[2] = nowNs() - codegenStart;
  output(mlirStringRefCreate(LLVMGetBufferStart(buffer), LLVMGetBufferSize(buffer)), data);
  return mlirLogicalResultSuccess();
}
} // namespace
MlirLogicalResult beaverTranslateModuleToLLVMIRText(MlirOperation module, MlirStringCallback callback, void *data) {
  if (!module.ptr || !callback) return mlirLogicalResultFailure();
  Compilation c;
  c.module = mlirTranslateModuleToLLVMIR(module, c.context);
  if (!c.module) return mlirLogicalResultFailure();
  char *message = LLVMPrintModuleToString(c.module);
  if (!message) return mlirLogicalResultFailure();
  emitString(callback, data, message);
  LLVMDisposeMessage(message);
  return mlirLogicalResultSuccess();
}
MlirLogicalResult beaverCompileLLVMIRToPTX(MlirStringRef ir, MlirStringRef cpu, MlirStringRef features,
    MlirStringCallback output, MlirStringCallback error, void *data) {
  if (!ir.data) return mlirLogicalResultFailure();
  initializeTargets();
  return compile({nullptr}, ir, {"nvptx64-nvidia-cuda", unwrap(cpu).str(), unwrap(features).str()},
                 0, LLVMCodeGenLevelDefault, false, true, nullptr, output, error, data);
}
extern "C" MLIR_CAPI_EXPORTED void beaverLLVMGetHostTarget(MlirStringCallback callback, void *data) {
  if (callback) for (const auto &value : hostTarget()) emitString(callback, data, value.c_str());
}
extern "C" MLIR_CAPI_EXPORTED MlirLogicalResult beaverCompileToObject(
    MlirOperation module, MlirStringRef ir, MlirStringRef triple, MlirStringRef cpu, MlirStringRef features,
    unsigned level, bool pic, bool assembly, uint64_t *timings,
    MlirStringCallback output, MlirStringCallback error, void *data) {
  auto target = hostTarget();
  if (triple.length && unwrap(triple) != target[0]) target = {unwrap(triple).str(), "", ""};
  if (cpu.length) target[1] = unwrap(cpu).str();
  if (features.length) target[2] = unwrap(features).str();
  auto cgOpt = static_cast<LLVMCodeGenOptLevel>(std::min(level, 3u));
  return compile(module, ir, target, level, cgOpt, pic, assembly, timings, output, error, data);
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
    if (auto fnOp = llvm::dyn_cast<mlir::FunctionOpInterface>(&op)) {
      if (fnOp.isExternal()) continue;
      bool selected = fnOp.getName() == targetName;
      found |= selected;
      if (!selected) {
        fnOp.getFunctionBody().dropAllReferences();
        fnOp.getFunctionBody().getBlocks().clear();
      }
      if (auto llvmFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(&op)) {
        if (!selected || llvmFunc.getLinkage() == mlir::LLVM::Linkage::Internal ||
                         llvmFunc.getLinkage() == mlir::LLVM::Linkage::Private)
          llvmFunc.setLinkage(mlir::LLVM::Linkage::External);
      } else if (!selected) {
        mlir::SymbolTable::setSymbolVisibility(&op, mlir::SymbolTable::Visibility::Private);
      }
    } else if (!keepGlobals) {
      if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(&op)) {
        if (globalOp.getLinkage() == mlir::LLVM::Linkage::Internal ||
            globalOp.getLinkage() == mlir::LLVM::Linkage::Private) { continue; }
        globalOp.removeValueAttr();
        globalOp.getInitializerRegion().dropAllReferences();
        globalOp.getInitializerRegion().getBlocks().clear();
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
