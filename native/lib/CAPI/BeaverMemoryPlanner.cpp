#include "mlir-c/Beaver/Pass.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace {
// The upstream planner does not free allocated arenas or check argument size.
// Keep planning upstream; supply these ownership/bounds checks at the boundary.
struct StaticMemoryPlan : PassWrapper<StaticMemoryPlan, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticMemoryPlan)
  StaticMemoryPlan(bool callerArena, bool bestFit) : callerArena(callerArena), bestFit(bestFit) {}
  StringRef getArgument() const override { return "beaver-static-memory-plan"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) return;
    MemRefType arenaType;
    if (callerArena) {
      if (func.getNumArguments()) arenaType = dyn_cast<MemRefType>(func.getArgument(0).getType());
      if (!arenaType || arenaType.getRank() != 1 || !arenaType.getElementType().isInteger(8) ||
          !arenaType.hasStaticShape() || !arenaType.getLayout().isIdentity()) {
        func.emitError("caller arena must be a static, identity-layout memref<Nxi8> first argument");
        return signalPassFailure();
      }
    }
    llvm::SmallPtrSet<Operation *, 16> originalAllocs, originalViews;
    bool unsupportedStorage = false;
    func.walk([&](memref::AllocOp alloc) {
      originalAllocs.insert(alloc);
      auto type = alloc.getType();
      if (!type.hasStaticShape() || !type.getLayout().isIdentity() || !type.getElementType().isIntOrFloat()) return;
      unsigned bits = type.getElementTypeBitWidth();
      if (bits < 8 || bits > 128 || (bits & (bits - 1))) {
        alloc.emitError("static memory planning requires power-of-two element storage widths from 8 to 128 bits");
        unsupportedStorage = true;
      }
    });
    if (unsupportedStorage) return signalPassFailure();
    func.walk([&](memref::ViewOp view) { originalViews.insert(view); });
    bufferization::StaticMemoryPlannerAnalysisPassOptions options;
    options.arenaMode = callerArena ? "arg" : "allocate";
    options.algorithm = bestFit ? bufferization::MemoryPlannerAlgorithm::BestFit : bufferization::MemoryPlannerAlgorithm::Trivial;
    OpPassManager pipeline(func::FuncOp::getOperationName());
    pipeline.addPass(bufferization::createStaticMemoryPlannerAnalysisPass(options));
    if (failed(runPipeline(pipeline, func))) return signalPassFailure();
    if (callerArena) {
      func.walk([&](memref::ViewOp view) {
        if (originalViews.contains(view) || view.getSource() != func.getArgument(0)) return;
        auto offset = view.getByteShift().getDefiningOp<arith::ConstantIndexOp>();
        auto type = view.getType();
        int64_t bytes = type.getNumElements() * ((type.getElementTypeBitWidth() + 7) / 8);
        if (offset.value() > arenaType.getDimSize(0) || bytes > arenaType.getDimSize(0) - offset.value()) {
          view.emitError("caller arena is too small for the planned buffers");
          signalPassFailure();
        }
      });
    } else {
      SmallVector<Value> arenas;
      func.walk([&](memref::AllocOp alloc) { if (!originalAllocs.contains(alloc)) arenas.push_back(alloc); });
      for (Block &block : func.getBody()) {
        if (!isa<func::ReturnOp>(block.getTerminator())) continue;
        OpBuilder builder(block.getTerminator());
        for (Value arena : arenas) memref::DeallocOp::create(builder, func.getLoc(), arena);
      }
    }
  }
  bool callerArena, bestFit;
};
} // namespace

MLIR_CAPI_EXPORTED MlirPass beaverCreateStaticMemoryPlannerPass(bool callerArena, bool bestFit) {
  return wrap(new StaticMemoryPlan(callerArena, bestFit));
}
