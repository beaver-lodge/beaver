#include "mlir-c/Beaver/Triton.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "llvm/Support/ErrorHandling.h"

#ifdef BEAVER_ENABLE_TRITON
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

void registerTritonConversionInterfaces(DialectRegistry &registry) {
  LLVM::registerInlinerInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  arith::registerConvertArithToLLVMInterface(registry);
}

} // namespace

MLIR_CAPI_EXPORTED bool
beaverContextRegisterTritonDialects(MlirContext context) {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<
      triton::TritonDialect, gpu::TritonGPUDialect,
      nvidia_gpu::TritonNvidiaGPUDialect,
      instrument::TritonInstrumentDialect, gluon::GluonDialect>();
  registerTritonConversionInterfaces(registry);
  MLIRContext *ctx = unwrap(context);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  return true;
}

MLIR_CAPI_EXPORTED bool beaverRegisterTritonPasses() {
  registerTritonPasses();
  gpu::registerTritonGPUPasses();
  nvidia_gpu::registerTritonNvidiaGPUPasses();
  instrument::registerTritonInstrumentPasses();
  gluon::registerGluonPasses();
  registerConvertTritonToTritonGPUPass();
  registerRelayoutTritonGPUPass();
  gpu::registerAllocateSharedMemoryPass();
  gpu::registerTritonGPUAllocateWarpGroups();
  gpu::registerTritonGPUGlobalScratchAllocationPass();
  gpu::registerCanonicalizeLLVMIR();
  registerConvertWarpSpecializeToLLVM();
  registerInitializeWSClusterBarriers();
  registerConvertTritonGPUToLLVMPass();
  mlir::registerLLVMDIScope();
  mlir::registerLLVMDILocalVariable();
  return true;
}

#else

MLIR_CAPI_EXPORTED bool
beaverContextRegisterTritonDialects(MlirContext context) {
  return false;
}

MLIR_CAPI_EXPORTED bool beaverRegisterTritonPasses() { return false; }

#endif
