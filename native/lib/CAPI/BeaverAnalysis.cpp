#include "mlir-c/Beaver/Analysis.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ScopeExit.h"
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
