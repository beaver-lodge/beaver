#include "mlir-c/Beaver/Analysis.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
namespace {
struct Session {
  Operation *root;
  ModuleAnalysisManager analyses;
  uint64_t generation = 0;
  Session(Operation *root) : root(root), analyses(root, nullptr) {}
  bool contains(Operation *op) { return op && root->isAncestor(op); }
};
bool hasLayout(Type type, DenseSet<Type> &visiting) {
  if (!visiting.insert(type).second) return false;
  auto leave = llvm::scope_exit([&] { visiting.erase(type); });
  if (auto vector = dyn_cast<VectorType>(type))
    return vector.getRank() != 0 && hasLayout(vector.getElementType(), visiting);
  if (auto complex = dyn_cast<ComplexType>(type)) return hasLayout(complex.getElementType(), visiting);
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) return hasLayout(array.getElementType(), visiting);
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type))
    return !structure.isOpaque() && structure.isInitialized() &&
        llvm::all_of(structure.getBody(), [&](Type element) { return hasLayout(element, visiting); });
  return type.isIntOrIndexOrFloat() || isa<DataLayoutTypeInterface>(type);
}
}
void *beaverAnalysisCreate(MlirOperation root) { return new Session(unwrap(root)); }
void beaverAnalysisDestroy(void *session) { delete static_cast<Session *>(session); }
void beaverAnalysisInvalidate(void *session, unsigned preserved) {
  auto &s = *static_cast<Session *>(session);
  AnalysisManager::PreservedAnalyses pa;
  if (preserved & 1) pa.preserve<DominanceInfo>();
  if (preserved & 2) pa.preserve<PostDominanceInfo>();
  if (preserved & 4) pa.preserve<Liveness>();
  if (preserved & 8) pa.preserve<DataLayoutAnalysis>();
  AnalysisManager(s.analyses).invalidate(pa);
  ++s.generation;
}
uint64_t beaverAnalysisGeneration(void *session) { return static_cast<Session *>(session)->generation; }
unsigned beaverAnalysisCached(void *session) {
  auto &s = *static_cast<Session *>(session);
  AnalysisManager am(s.analyses);
  return unsigned(bool(am.getCachedAnalysis<DominanceInfo>())) |
      unsigned(bool(am.getCachedAnalysis<PostDominanceInfo>())) << 1 |
      unsigned(bool(am.getCachedAnalysis<Liveness>())) << 2 |
      unsigned(bool(am.getCachedAnalysis<DataLayoutAnalysis>())) << 3;
}
int beaverAnalysisQuery(void *session, unsigned kind, MlirOperation lhs,
                        MlirOperation rhs, MlirValue value) {
  auto &s = *static_cast<Session *>(session);
  auto *a = unwrap(lhs);
  auto *b = unwrap(rhs);
  if (!s.contains(a)) return -1;
  AnalysisManager am(s.analyses);
  if (kind < 2) {
    if (!s.contains(b)) return -1;
    return kind == 0 ? am.getAnalysis<DominanceInfo>().dominates(a, b)
                     : am.getAnalysis<PostDominanceInfo>().postDominates(a, b);
  }
  if (kind != 2 || !value.ptr) return -1;
  auto v = unwrap(value);
  auto *owner = v.getDefiningOp();
  if (!owner) owner = cast<BlockArgument>(v).getOwner()->getParentOp();
  if (!s.contains(owner)) return -1;
  return am.getAnalysis<Liveness>().isDeadAfter(v, a);
}
bool beaverAnalysisLayout(void *session, MlirOperation scope, MlirType type, uint64_t *out) {
  auto &s = *static_cast<Session *>(session);
  auto *op = unwrap(scope);
  auto t = unwrap(type);
  if (!t || !s.contains(op) || t.getContext() != s.root->getContext()) return false;
  // Reject unsupported types before upstream's fatal missing-layout diagnostic.
  DenseSet<Type> visiting;
  if (!hasLayout(t, visiting)) return false;
  const auto &layout = AnalysisManager(s.analyses).getAnalysis<DataLayoutAnalysis>().getAtOrAbove(op);
  auto size = layout.getTypeSize(t), bits = layout.getTypeSizeInBits(t), alloc = layout.getTypeAllocSize(t);
  out[0] = size.getKnownMinValue(); out[1] = size.isScalable();
  out[2] = bits.getKnownMinValue(); out[3] = bits.isScalable();
  out[4] = alloc.getKnownMinValue(); out[5] = alloc.isScalable();
  out[6] = layout.getTypeABIAlignment(t); out[7] = layout.getTypePreferredAlignment(t);
  out[8] = layout.getTypeIndexBitwidth(t).value_or(0);
  return true;
}

namespace {
template <bool Shaped, typename Callback>
MlirLogicalResult infer(MlirStringRef name, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    MlirAttribute properties, intptr_t nRegions, MlirRegion *regions, Callback callback, void *data) {
  auto *ctx = unwrap(context);
  auto loc = unwrap(location);
  auto info = RegisteredOperationName::lookup(unwrap(name), ctx);
  using Interface = std::conditional_t<Shaped, InferShapedTypeOpInterface, InferTypeOpInterface>;
  if (!info || !info->hasInterface<Interface>() || loc.getContext() != ctx) return wrap(failure());
  auto attrs = dyn_cast_or_null<DictionaryAttr>(unwrap(attributes));
  if (!attrs || attrs.getContext() != ctx) return wrap(failure());
  SmallVector<Value> values;
  for (intptr_t i = 0; i < nOperands; ++i) {
    if (!operands[i].ptr || unwrap(operands[i]).getContext() != ctx) return wrap(failure());
    values.push_back(unwrap(operands[i]));
  }
  SmallVector<Region *> borrowed;
  for (intptr_t i = 0; i < nRegions; ++i) {
    if (!regions[i].ptr) return wrap(failure());
    auto *region = unwrap(regions[i]);
    bool valid = true;
    region->walk([&](Operation *op) { valid &= op->getContext() == ctx; });
    for (Block &block : *region)
      for (BlockArgument arg : block.getArguments()) valid &= arg.getContext() == ctx;
    if (!valid) return wrap(failure());
    borrowed.push_back(region);
  }
  PropertyRef props;
  std::unique_ptr<char[]> storage;
  auto cleanup = llvm::scope_exit([&] { if (props) info->destroyOpProperties(props); });
  if (properties.ptr) {
    auto attr = unwrap(properties);
    if (attr.getContext() != ctx || !info->getOpPropertyByteSize()) return wrap(failure());
    if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
      NamedAttrList inherent(dict);
      if (failed(info->verifyInherentAttrs(inherent, [&] { return emitError(loc); }))) return wrap(failure());
    }
    storage = std::make_unique<char[]>(info->getOpPropertyByteSize());
    props = PropertyRef(info->getOpPropertiesTypeID(), storage.get());
    info->initOpProperties(props, {});
    info->populateDefaultProperties(props);
    if (failed(info->setOpPropertiesFromAttribute(*info, props, attr, [&] { return emitError(loc); })))
      return wrap(failure());
  }
  if constexpr (Shaped) {
    SmallVector<ShapedTypeComponents> result;
    if (failed(info->getInterface<Interface>()->inferReturnTypeComponents(ctx, loc, ValueRange(values), attrs, props, borrowed, result)))
      return wrap(failure());
    for (const auto &part : result)
      callback(part.hasRank(), part.hasRank() ? part.getDims().size() : 0,
          part.hasRank() ? part.getDims().data() : nullptr, wrap(part.getElementType()), wrap(part.getAttribute()), data);
  } else {
    SmallVector<Type> result;
    if (failed(info->getInterface<Interface>()->inferReturnTypes(ctx, loc, values, attrs, props, borrowed, result)))
      return wrap(failure());
    SmallVector<MlirType> wrapped;
    for (Type type : result) wrapped.push_back(wrap(type));
    callback(wrapped.size(), wrapped.data(), data);
  }
  return wrap(success());
}
}
#define INFER_ARGS MlirStringRef name, MlirContext ctx, MlirLocation loc, intptr_t nv, MlirValue *vs, MlirAttribute attrs, MlirAttribute props, intptr_t nr, MlirRegion *rs
MlirLogicalResult beaverInferReturnTypes(INFER_ARGS, MlirTypesCallback cb, void *data) {
  return infer<false>(name, ctx, loc, nv, vs, attrs, props, nr, rs, cb, data);
}
MlirLogicalResult beaverInferReturnComponents(INFER_ARGS, MlirShapedTypeComponentsCallback cb, void *data) {
  return infer<true>(name, ctx, loc, nv, vs, attrs, props, nr, rs, cb, data);
}
#undef INFER_ARGS

MlirLogicalResult beaverConversionRun(unsigned mode, MlirOperation root,
    MlirConversionTarget target, MlirFrozenRewritePatternSet patterns,
    MlirConversionConfig config, MlirStringCallback report, void *data) {
  struct Changes : RewriterBase::Listener {
    bool changed = false;
    void notifyOperationInserted(Operation *, OpBuilder::InsertPoint) override { changed = true; }
    void notifyOperationModified(Operation *) override { changed = true; }
    void notifyOperationErased(Operation *) override { changed = true; }
    void notifyBlockInserted(Block *, Region *, Region::iterator) override { changed = true; }
    void notifyBlockErased(Block *) override { changed = true; }
  } changes;
  // Snapshot identities before conversion. Reports never expose handles that
  // a successful rewrite may have erased, or dereference rolled-back new ops.
  struct Identity { Operation *op; std::string name, location; };
  SmallVector<Identity> original;
  if (report) unwrap(root)->walk<WalkOrder::PreOrder>([&](Operation *op) {
    std::string location;
    llvm::raw_string_ostream(location) << op->getLoc();
    original.push_back({op, op->getName().getStringRef().str(), std::move(location)});
  });
  DenseSet<Operation *> legalizable, unlegalized;
  ConversionConfig options = *unwrap(config);
  options.allowPatternRollback = true;
  options.listener = &changes;
  if (mode == 1) options.unlegalizedOps = &unlegalized;
  if (mode == 2) options.legalizableOps = &legalizable;
  LogicalResult result = failure();
  if (mode == 0) result = applyFullConversion(unwrap(root), *unwrap(target), *unwrap(patterns), options);
  if (mode == 1) result = applyPartialConversion(unwrap(root), *unwrap(target), *unwrap(patterns), options);
  if (mode == 2) result = applyAnalysisConversion(unwrap(root), *unwrap(target), *unwrap(patterns), options);
  if (report) {
    auto snapshots = [&](const DenseSet<Operation *> &selected) {
      llvm::json::Array out;
      for (auto [index, entry] : llvm::enumerate(original))
        if (selected.contains(entry.op))
          out.push_back(llvm::json::Object{{"index", int64_t(index)}, {"name", entry.name}, {"location", entry.location}});
      return out;
    };
    llvm::json::Object object{{"success", succeeded(result)}, {"changed", mode != 2 && succeeded(result) && changes.changed},
      {"rolled_back", mode == 2 || failed(result)}, {"legalizable", snapshots(legalizable)},
      {"unlegalized", snapshots(unlegalized)}, {"unlegalized_complete", mode == 1 && succeeded(result)}};
    std::string json;
    llvm::raw_string_ostream(json) << llvm::json::Value(std::move(object));
    report(wrap(StringRef(json)), data);
  }
  return wrap(result);
}

extern "C" MLIR_CAPI_EXPORTED MlirType beaverConvertCompositeType(
    MlirType type, const MlirType *sources, size_t numSources, MlirType target) {
  if (!type.ptr || !target.ptr || numSources == 0) return type;
  AttrTypeReplacer replacer;
  Type targetType = unwrap(target);
  replacer.addReplacement([sources, numSources, targetType](Type t) -> std::optional<Type> {
    for (size_t i = 0; i < numSources; ++i) {
      if (t == unwrap(sources[i])) return targetType;
    }
    return std::nullopt;
  });
  return wrap(replacer.replace(unwrap(type)));
}

namespace {
struct FuncRegionSignatureConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FunctionType type = op.getFunctionType();
    TypeConverter::SignatureConversion result(type.getNumInputs());
    if (failed(getTypeConverter()->convertSignatureArgs(type.getInputs(), result)))
      return failure();
    SmallVector<Type, 1> newResults;
    if (failed(getTypeConverter()->convertTypes(type.getResults(), newResults)))
      return failure();
    if (!op.getBody().empty()) {
      if (failed(rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter(), &result)))
        return failure();
    }
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(FunctionType::get(op.getContext(), result.getConvertedTypes(), newResults));
    });
    return success();
  }
};
} // namespace

extern "C" MLIR_CAPI_EXPORTED void beaverPopulateStructuralTypeConversions(
    MlirConversionTarget target, MlirRewritePatternSet patterns,
    MlirTypeConverter typeConverter, unsigned domainMask) {
  ConversionTarget &convTarget = *unwrap(target);
  RewritePatternSet &patternSet = *unwrap(patterns);
  TypeConverter *tcPtr = unwrap(typeConverter);
  TypeConverter &tc = *tcPtr;

  if (domainMask & 1) {
    patternSet.add<FuncRegionSignatureConversion>(tc, patternSet.getContext(), 2);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patternSet, tc);
    populateCallOpTypeConversionPattern(patternSet, tc);
    populateReturnOpTypeConversionPattern(patternSet, tc);
    convTarget.addDynamicallyLegalOp<func::FuncOp>([tcPtr](func::FuncOp op) {
      return tcPtr->isSignatureLegal(op.getFunctionType()) && tcPtr->isLegal(&op.getBody());
    });
    convTarget.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([tcPtr](LLVM::LLVMFuncOp op) {
      return tcPtr->isLegal(op.getFunctionType()) && tcPtr->isLegal(&op.getBody());
    });
    convTarget.addDynamicallyLegalOp<func::CallOp>([tcPtr](func::CallOp op) { return tcPtr->isLegal(op); });
    convTarget.addDynamicallyLegalOp<func::ReturnOp>([tcPtr](func::ReturnOp op) { return tcPtr->isLegal(op); });
    convTarget.addDynamicallyLegalOp<LLVM::ReturnOp>([tcPtr](LLVM::ReturnOp op) { return tcPtr->isLegal(op); });
  }
  if (domainMask & 2) {
    populateBranchOpInterfaceTypeConversionPattern(patternSet, tc);
    cf::populateCFStructuralTypeConversionsAndLegality(tc, patternSet, convTarget);
  }
  if (domainMask & 4) {
    scf::populateSCFStructuralTypeConversionsAndLegality(tc, patternSet, convTarget);
  }
}

namespace {
struct SafepointColoring {
  SmallVector<Type> slotTypes;
  SmallVector<Value> coloredValues;
  SmallVector<size_t> coloredValueSlots;
  SmallVector<SmallVector<Value>> liveAt;
  SmallVector<SmallVector<size_t>> slotsAt;
};

bool computeSafepointColoring(Operation *func, ArrayRef<Operation *> safepoints,
                              DominanceInfo &domInfo, Liveness &liveness,
                              bool excludeEntryArgs, SafepointColoring &out) {
  if (!func || func->getNumRegions() == 0 || func->getRegion(0).empty()) return false;
  Block *entryBlock = &func->getRegion(0).front();
  for (Operation *sp : safepoints) {
    if (!sp || !func->isProperAncestor(sp)) return false;
  }

  SmallVector<Value> candidates;
  func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == func) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          if (excludeEntryArgs && &block == entryBlock) continue;
          for (BlockArgument arg : block.getArguments()) candidates.push_back(arg);
        }
      }
      return;
    }
    for (Value res : op->getResults()) candidates.push_back(res);
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) candidates.push_back(arg);
      }
    }
  });

  size_t numSp = safepoints.size();
  out.liveAt.resize(numSp);
  out.slotsAt.resize(numSp);
  SmallVector<llvm::SmallBitVector> slotBusy;

  for (Value v : candidates) {
    llvm::SmallBitVector liveMask(numSp, false);
    for (size_t i = 0; i < numSp; ++i) {
      Operation *sp = safepoints[i];
      bool dominates = false;
      if (Operation *defOp = v.getDefiningOp()) {
        dominates = (defOp != sp) && domInfo.properlyDominates(defOp, sp);
      } else {
        Block *defBlock = cast<BlockArgument>(v).getOwner();
        dominates = domInfo.dominates(defBlock, sp->getBlock());
      }
      if (dominates && !liveness.isDeadAfter(v, sp)) {
        liveMask.set(i);
      }
    }
    if (!liveMask.any()) continue;

    size_t assignedSlot = SIZE_MAX;
    for (size_t s = 0; s < out.slotTypes.size(); ++s) {
      if (out.slotTypes[s] == v.getType() && !slotBusy[s].anyCommon(liveMask)) {
        assignedSlot = s;
        slotBusy[s] |= liveMask;
        break;
      }
    }
    if (assignedSlot == SIZE_MAX) {
      assignedSlot = out.slotTypes.size();
      out.slotTypes.push_back(v.getType());
      slotBusy.push_back(liveMask);
    }
    out.coloredValues.push_back(v);
    out.coloredValueSlots.push_back(assignedSlot);
    for (size_t i = 0; i < numSp; ++i) {
      if (liveMask.test(i)) {
        out.liveAt[i].push_back(v);
        out.slotsAt[i].push_back(assignedSlot);
      }
    }
  }
  return true;
}
} // namespace

using BeaverSafepointCallback = void (*)(size_t totalSlots, const MlirType *slotTypes,
                                         size_t numColoredValues, const MlirValue *coloredValues,
                                         const size_t *valueSlots, size_t safepointIndex,
                                         MlirOperation safepointOp, size_t numLive,
                                         const MlirValue *liveValues, const size_t *liveSlots,
                                         void *userData);

extern "C" MLIR_CAPI_EXPORTED bool beaverAnalyzeSafepoints(
    void *session, MlirOperation funcOp, const MlirOperation *safepointOps,
    size_t numSafepoints, bool excludeEntryArgs, BeaverSafepointCallback callback,
    void *userData) {
  Operation *func = unwrap(funcOp);
  if (!func || !callback) return false;
  SmallVector<Operation *> sps;
  for (size_t i = 0; i < numSafepoints; ++i) sps.push_back(unwrap(safepointOps[i]));

  SafepointColoring coloring;
  if (session) {
    auto &s = *static_cast<Session *>(session);
    if (!s.contains(func)) return false;
    AnalysisManager am(s.analyses);
    auto &dom = am.getAnalysis<DominanceInfo>();
    auto &live = am.getAnalysis<Liveness>();
    if (!computeSafepointColoring(func, sps, dom, live, excludeEntryArgs, coloring)) return false;
  } else {
    DominanceInfo dom(func);
    Liveness live(func);
    if (!computeSafepointColoring(func, sps, dom, live, excludeEntryArgs, coloring)) return false;
  }

  SmallVector<MlirType> wrappedSlotTypes;
  for (Type t : coloring.slotTypes) wrappedSlotTypes.push_back(wrap(t));
  SmallVector<MlirValue> wrappedColoredValues;
  for (Value v : coloring.coloredValues) wrappedColoredValues.push_back(wrap(v));

  if (numSafepoints == 0) {
    callback(0, nullptr, 0, nullptr, nullptr, SIZE_MAX, {nullptr}, 0, nullptr, nullptr, userData);
    return true;
  }

  for (size_t i = 0; i < numSafepoints; ++i) {
    SmallVector<MlirValue> wrappedLive;
    for (Value v : coloring.liveAt[i]) wrappedLive.push_back(wrap(v));
    callback(wrappedSlotTypes.size(), wrappedSlotTypes.data(),
             wrappedColoredValues.size(), wrappedColoredValues.data(),
             coloring.coloredValueSlots.data(), i, safepointOps[i],
             wrappedLive.size(), wrappedLive.data(), coloring.slotsAt[i].data(),
             userData);
  }
  return true;
}

extern "C" MLIR_CAPI_EXPORTED bool beaverLowerResumableCFG(
    MlirOperation funcOp, const MlirOperation *safepointOps, size_t numSafepoints,
    MlirStringRef saveOpName, MlirStringRef loadOpName, intptr_t stateArgIndex) {
  Operation *func = unwrap(funcOp);
  if (!func || func->getNumRegions() == 0 || func->getRegion(0).empty()) return false;
  if (numSafepoints == 0) return true;

  SmallVector<Operation *> sps;
  for (size_t i = 0; i < numSafepoints; ++i) {
    Operation *sp = unwrap(safepointOps[i]);
    if (!sp || !func->isProperAncestor(sp)) return false;
    sps.push_back(sp);
  }

  DominanceInfo initialDom(func);
  Liveness initialLiveness(func);
  SafepointColoring coloring;
  if (!computeSafepointColoring(func, sps, initialDom, initialLiveness, true, coloring))
    return false;

  Location funcLoc = func->getLoc();
  Region &body = func->getRegion(0);

  // 1. Insert save operations immediately before each safepoint.
  for (size_t i = 0; i < numSafepoints; ++i) {
    Operation *sp = sps[i];
    OpBuilder builder(sp);
    for (size_t k = 0; k < coloring.liveAt[i].size(); ++k) {
      OperationState saveState(sp->getLoc(), unwrap(saveOpName));
      saveState.addOperands(coloring.liveAt[i][k]);
      saveState.addAttribute("slot", builder.getI64IntegerAttr(coloring.slotsAt[i][k]));
      saveState.addAttribute("safepoint", builder.getI64IntegerAttr(i + 1));
      builder.create(saveState);
    }
  }

  // 2. Split each safepoint block right after the safepoint op, keeping a temporary
  //    branch from preBlock to resumeBlock so dominance reflects the linear order.
  SmallVector<Block *> preBlocks(numSafepoints);
  SmallVector<Block *> resumeBlocks(numSafepoints);
  for (size_t i = 0; i < numSafepoints; ++i) {
    Operation *sp = sps[i];
    Block *preBlock = sp->getBlock();
    Block *resumeBlock = preBlock->splitBlock(std::next(Block::iterator(sp)));
    preBlocks[i] = preBlock;
    resumeBlocks[i] = resumeBlock;
    OpBuilder builder = OpBuilder::atBlockEnd(preBlock);
    cf::BranchOp::create(builder, sp->getLoc(), resumeBlock);
  }

  // 3. Insert reload operations at the start of each resumeBlock and rewrite
  //    dominated uses from the deepest safepoint back to the first.
  DominanceInfo splitDom(func);
  SmallVector<SmallVector<Value>> reloadedAt(numSafepoints);
  for (size_t i = 0; i < numSafepoints; ++i) {
    OpBuilder builder = OpBuilder::atBlockBegin(resumeBlocks[i]);
    Location loc = sps[i]->getLoc();
    for (size_t k = 0; k < coloring.liveAt[i].size(); ++k) {
      Value origVal = coloring.liveAt[i][k];
      OperationState loadState(loc, unwrap(loadOpName));
      loadState.addTypes(origVal.getType());
      loadState.addAttribute("slot", builder.getI64IntegerAttr(coloring.slotsAt[i][k]));
      loadState.addAttribute("safepoint", builder.getI64IntegerAttr(i + 1));
      Operation *loadOp = builder.create(loadState);
      reloadedAt[i].push_back(loadOp->getResult(0));
    }
  }

  for (size_t rev = numSafepoints; rev > 0; --rev) {
    size_t i = rev - 1;
    Block *resumeBlock = resumeBlocks[i];
    for (size_t k = 0; k < coloring.liveAt[i].size(); ++k) {
      Value origVal = coloring.liveAt[i][k];
      Value reloaded = reloadedAt[i][k];
      origVal.replaceUsesWithIf(reloaded, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        if (user == reloaded.getDefiningOp()) return false;
        return splitDom.dominates(resumeBlock, user->getBlock());
      });
    }
  }

  // 4. Replace the temporary branch + safepoint in each preBlock with a suspend return.
  SmallVector<Type> funcResTypes;
  if (auto fnOp = dyn_cast<FunctionOpInterface>(func)) {
    llvm::append_range(funcResTypes, fnOp.getResultTypes());
  }
  for (size_t i = 0; i < numSafepoints; ++i) {
    Operation *sp = sps[i];
    Location loc = sp->getLoc();
    Operation *tempBr = preBlocks[i]->getTerminator();
    OpBuilder builder(tempBr);
    if (funcResTypes.size() == 1 && isa<IntegerType, IndexType>(funcResTypes[0])) {
      Type retType = funcResTypes[0];
      Attribute stateAttr = retType.isIndex()
                                ? static_cast<Attribute>(builder.getIndexAttr(i + 1))
                                : static_cast<Attribute>(builder.getIntegerAttr(retType, i + 1));
      Value stateVal = arith::ConstantOp::create(builder, loc, retType, cast<TypedAttr>(stateAttr));
      func::ReturnOp::create(builder, loc, ValueRange{stateVal});
    } else if (funcResTypes.empty()) {
      func::ReturnOp::create(builder, loc);
    } else {
      return false;
    }
    tempBr->erase();
    if (sp->use_empty()) sp->erase();
  }

  // 5. Create entry dispatch block with cf.switch jumping to originalEntry or resumeBlocks.
  Block *originalEntry = &body.front();
  Block *dispatchBlock = new Block();
  body.getBlocks().insert(body.begin(), dispatchBlock);
  for (BlockArgument arg : originalEntry->getArguments()) {
    BlockArgument newArg = dispatchBlock->addArgument(arg.getType(), arg.getLoc());
    arg.replaceAllUsesWith(newArg);
  }
  while (originalEntry->getNumArguments() > 0) {
    originalEntry->eraseArgument(0);
  }

  OpBuilder entryBuilder = OpBuilder::atBlockBegin(dispatchBlock);
  Value stateValue;
  if (stateArgIndex >= 0 && static_cast<size_t>(stateArgIndex) < dispatchBlock->getNumArguments()) {
    stateValue = dispatchBlock->getArgument(stateArgIndex);
    if (stateValue.getType().isIndex()) {
      stateValue = arith::IndexCastOp::create(entryBuilder, funcLoc, entryBuilder.getI32Type(), stateValue);
    }
  } else {
    OperationState stateOp(funcLoc, "coro.state");
    stateOp.addTypes(entryBuilder.getI32Type());
    stateValue = entryBuilder.create(stateOp)->getResult(0);
  }

  IntegerType switchIntType = dyn_cast<IntegerType>(stateValue.getType());
  if (!switchIntType) return false;

  SmallVector<APInt> caseValues;
  SmallVector<Block *> caseDestinations;
  SmallVector<ValueRange> caseOperands;
  for (size_t i = 0; i < numSafepoints; ++i) {
    caseValues.push_back(APInt(switchIntType.getWidth(), i + 1));
    caseDestinations.push_back(resumeBlocks[i]);
    caseOperands.push_back(ValueRange{});
  }
  cf::SwitchOp::create(entryBuilder, funcLoc, stateValue, originalEntry, ValueRange{},
                       caseValues, caseDestinations, caseOperands);
  return true;
}
