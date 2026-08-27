#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/IRMapping.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir-c/Beaver/Conversion.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ExtensibleDialect.h"
#ifdef BEAVER_HAS_MLIR_COMPOSITE_FAILURE_ACTION
#include "mlir/Transforms/CompositePass.h"
#endif
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/VCSRevision.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;

namespace {

static bool isExTermType(Type type) {
  auto dynamicType = dyn_cast<DynamicType>(type);
  if (!dynamicType)
    return false;

  StringRef name = dynamicType.getTypeDef()->getName();
  return name == "term" || name == "bound" || name == "unbound" ||
         name.ends_with(".term") || name.ends_with(".bound") ||
         name.ends_with(".unbound");
}

static Operation *exRuntimeSymbolTable(Operation *operation) {
  Operation *scope = operation->getParentOp();
  while (scope && !scope->hasTrait<OpTrait::SymbolTable>())
    scope = scope->getParentOp();
  return scope;
}

static LogicalResult ensureExRuntimeDeclaration(
    Operation *operation, ConversionPatternRewriter &rewriter,
    StringRef symbol, unsigned arity) {
  Operation *symbolTable = exRuntimeSymbolTable(operation);
  if (!symbolTable || symbolTable->getNumRegions() != 1 ||
      symbolTable->getRegion(0).empty())
    return failure();
  if (SymbolTable::lookupSymbolIn(symbolTable, symbol))
    return success();

  SmallVector<Type> inputs(arity, rewriter.getI64Type());
  SmallVector<Type> results{rewriter.getI64Type()};
  FunctionType functionType = rewriter.getFunctionType(inputs, results);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&symbolTable->getRegion(0).front());
  OperationState state(operation->getLoc(), "func.func");
  state.addAttribute("sym_name", rewriter.getStringAttr(symbol));
  state.addAttribute("sym_visibility", rewriter.getStringAttr("private"));
  state.addAttribute("function_type", TypeAttr::get(functionType));
  state.addRegion();
  rewriter.create(state);
  return success();
}

static Value createExRuntimeCall(Operation *operation,
                                 ConversionPatternRewriter &rewriter,
                                 StringRef symbol, ValueRange operands) {
  OperationState state(operation->getLoc(), "func.call");
  state.addOperands(operands);
  state.addAttribute(
      "callee", FlatSymbolRefAttr::get(rewriter.getContext(), symbol));
  state.addTypes(rewriter.getI64Type());
  return rewriter.create(state)->getResult(0);
}

class ExLiteralConversion final : public ConversionPattern {
public:
  ExLiteralConversion(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "ex.lit", PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!operands.empty() || operation->getNumResults() != 1)
      return failure();

    Attribute value = operation->getAttr("value");
    Type resultType =
        getTypeConverter()->convertType(operation->getResult(0).getType());
    if (!value || !resultType)
      return failure();

    OperationState state(operation->getLoc(), "arith.constant");
    state.addAttribute("value", value);
    state.addTypes(resultType);
    Operation *constant = rewriter.create(state);
    rewriter.replaceOp(operation, constant->getResults());
    return success();
  }
};

class ExBoxConversion final : public ConversionPattern {
public:
  ExBoxConversion(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "ex.box", PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1 || operation->getNumOperands() != 1 ||
        operation->getNumResults() != 1)
      return failure();

    Value word = operands.front();
    if (!isExTermType(operation->getOperand(0).getType())) {
      IntegerType wordType = rewriter.getI64Type();

      OperationState constantState(operation->getLoc(), "arith.constant");
      constantState.addAttribute("value", rewriter.getI64IntegerAttr(3));
      constantState.addTypes(wordType);
      Operation *constant = rewriter.create(constantState);

      OperationState shiftState(operation->getLoc(), "arith.shli");
      shiftState.addOperands({word, constant->getResult(0)});
      shiftState.addTypes(wordType);
      word = rewriter.create(shiftState)->getResult(0);
    }

    rewriter.replaceOp(operation, word);
    return success();
  }
};

class ExIdentityConversion final : public ConversionPattern {
public:
  ExIdentityConversion(const TypeConverter &converter, StringRef rootName,
                       MLIRContext *context)
      : ConversionPattern(converter, rootName, PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1 || operation->getNumResults() != 1)
      return failure();
    rewriter.replaceOp(operation, operands.front());
    return success();
  }
};

class ExBinaryConversion final : public ConversionPattern {
public:
  ExBinaryConversion(const TypeConverter &converter, StringRef rootName,
                     StringRef targetName, MLIRContext *context)
      : ConversionPattern(converter, rootName, PatternBenefit(1), context),
        targetName(targetName.str()) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2 || operation->getNumResults() != 1)
      return failure();

    Type resultType =
        getTypeConverter()->convertType(operation->getResult(0).getType());
    if (!resultType)
      return failure();

    OperationState state(operation->getLoc(), targetName);
    state.addOperands(operands);
    state.addTypes(resultType);
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(operation, replacement->getResults());
    return success();
  }

private:
  std::string targetName;
};

static bool exCmpPredicateCode(StringRef predicate, int64_t &code) {
  if (predicate == "eq")
    code = 0;
  else if (predicate == "ne")
    code = 1;
  else if (predicate == "slt")
    code = 2;
  else if (predicate == "sle")
    code = 3;
  else if (predicate == "sgt")
    code = 4;
  else if (predicate == "sge")
    code = 5;
  else if (predicate == "ult")
    code = 6;
  else if (predicate == "ule")
    code = 7;
  else if (predicate == "ugt")
    code = 8;
  else if (predicate == "uge")
    code = 9;
  else
    return false;
  return true;
}

class ExCmpConversion final : public ConversionPattern {
public:
  ExCmpConversion(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "ex.cmp", PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2 || operation->getNumResults() != 1)
      return failure();

    StringAttr predicate = operation->getAttrOfType<StringAttr>("predicate");
    Type resultType =
        getTypeConverter()->convertType(operation->getResult(0).getType());
    int64_t predicateCode;
    if (!predicate || !resultType ||
        !exCmpPredicateCode(predicate.getValue(), predicateCode))
      return failure();

    OperationState compareState(operation->getLoc(), "arith.cmpi");
    compareState.addOperands(operands);
    compareState.addAttribute("predicate",
                              rewriter.getI64IntegerAttr(predicateCode));
    compareState.addTypes(rewriter.getI1Type());
    Operation *compare = rewriter.create(compareState);

    OperationState extendState(operation->getLoc(), "arith.extui");
    extendState.addOperands(compare->getResult(0));
    extendState.addTypes(resultType);
    Operation *extend = rewriter.create(extendState);

    rewriter.replaceOp(operation, extend->getResults());
    return success();
  }
};

class ExRuntimeCallConversion final : public ConversionPattern {
public:
  ExRuntimeCallConversion(const TypeConverter &converter, StringRef rootName,
                          StringRef symbol, unsigned arity,
                          MLIRContext *context)
      : ConversionPattern(converter, rootName, PatternBenefit(1), context),
        symbol(symbol.str()), arity(arity) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = operation->getNumResults() == 1
                          ? getTypeConverter()->convertType(
                                operation->getResult(0).getType())
                          : Type();
    if (operands.size() != arity || !resultType ||
        !resultType.isInteger(64) ||
        std::any_of(operands.begin(), operands.end(), [](Value value) {
          return !value.getType().isInteger(64);
        }) ||
        failed(ensureExRuntimeDeclaration(operation, rewriter, symbol, arity)))
      return failure();

    Value result = createExRuntimeCall(operation, rewriter, symbol, operands);
    rewriter.replaceOp(operation, result);
    return success();
  }

private:
  std::string symbol;
  unsigned arity;
};

class ExTermBinaryConversion final : public ConversionPattern {
public:
  ExTermBinaryConversion(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "ex.binary", PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = operation->getNumResults() == 1
                          ? getTypeConverter()->convertType(
                                operation->getResult(0).getType())
                          : Type();
    if (!resultType || !resultType.isInteger(64) ||
        std::any_of(operands.begin(), operands.end(), [](Value value) {
          return !value.getType().isInteger(64);
        }) ||
        failed(ensureExRuntimeDeclaration(operation, rewriter,
                                          "ex.term.list_cons", 2)) ||
        failed(ensureExRuntimeDeclaration(operation, rewriter,
                                          "ex.term.binary_from_list", 1)))
      return failure();

    OperationState nilState(operation->getLoc(), "arith.constant");
    nilState.addAttribute("value", rewriter.getI64IntegerAttr(1));
    nilState.addTypes(rewriter.getI64Type());
    Value tail = rewriter.create(nilState)->getResult(0);

    for (auto iterator = operands.rbegin(); iterator != operands.rend();
         ++iterator) {
      SmallVector<Value> arguments{*iterator, tail};
      tail = createExRuntimeCall(operation, rewriter, "ex.term.list_cons",
                                 arguments);
    }

    SmallVector<Value> arguments{tail};
    Value binary = createExRuntimeCall(
        operation, rewriter, "ex.term.binary_from_list", arguments);
    rewriter.replaceOp(operation, binary);
    return success();
  }
};

class ExIfConversion final : public ConversionPattern {
public:
  ExIfConversion(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "ex.if", PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1 || operation->getNumRegions() != 2)
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(operation->getResultTypes(),
                                                resultTypes)))
      return failure();

    OperationState truncateState(operation->getLoc(), "arith.trunci");
    truncateState.addOperands(operands.front());
    truncateState.addTypes(rewriter.getI1Type());
    Operation *condition = rewriter.create(truncateState);

    OperationState ifState(operation->getLoc(), "scf.if");
    ifState.addOperands(condition->getResult(0));
    ifState.addTypes(resultTypes);
    ifState.addRegion();
    ifState.addRegion();
    Operation *ifOperation = rewriter.create(ifState);
    ifOperation->getRegion(0).takeBody(operation->getRegion(0));
    ifOperation->getRegion(1).takeBody(operation->getRegion(1));

    rewriter.replaceOp(operation, ifOperation->getResults());
    return success();
  }
};

class ExYieldConversion final : public ConversionPattern {
public:
  ExYieldConversion(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "ex.yield", PatternBenefit(1), context) {}

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperationState state(operation->getLoc(), "scf.yield");
    state.addOperands(operands);
    Operation *yield = rewriter.create(state);
    rewriter.replaceOp(operation, yield);
    return success();
  }
};

} // namespace

void beaverPopulateExScalarConversionPatterns(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter) {
  RewritePatternSet *set = unwrap(patterns);
  TypeConverter *converter = unwrap(typeConverter);
  MLIRContext *context = set->getContext();

  set->add<ExLiteralConversion, ExBoxConversion, ExYieldConversion>(*converter,
                                                                    context);
  set->add<ExIdentityConversion>(*converter, "ex.to_word", context);
  set->add<ExIdentityConversion>(*converter, "ex.unbox", context);
  set->add<ExBinaryConversion>(*converter, "ex.add", "arith.addi", context);
  set->add<ExBinaryConversion>(*converter, "ex.sub", "arith.subi", context);
  set->add<ExBinaryConversion>(*converter, "ex.mul", "arith.muli", context);
  set->add<ExBinaryConversion>(*converter, "ex.div", "arith.divsi", context);
  set->add<ExBinaryConversion>(*converter, "ex.rem", "arith.remsi", context);
  set->add<ExCmpConversion, ExIfConversion>(*converter, context);
}

void beaverPopulateExRuntimeConversionPatterns(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter) {
  RewritePatternSet *set = unwrap(patterns);
  TypeConverter *converter = unwrap(typeConverter);
  MLIRContext *context = set->getContext();

  set->add<ExRuntimeCallConversion>(*converter, "ex.term_eq", "ex.term.eq", 2,
                                    context);
  set->add<ExRuntimeCallConversion>(*converter, "ex.binary_part",
                                    "ex.term.binary_part", 3, context);
  set->add<ExTermBinaryConversion>(*converter, context);
}

MLIR_CAPI_EXPORTED void
beaverOperationDestroyIterative(MlirOperation operation) {
  Operation *root = unwrap(operation);
  if (!root)
    return;

  llvm::SmallVector<Operation *> operations;
  llvm::SmallVector<Block *> blocks;
  llvm::SmallVector<Operation *> pending{root};

  while (!pending.empty()) {
    Operation *current = pending.pop_back_val();
    operations.push_back(current);

    for (Region &region : current->getRegions()) {
      for (Block &block : region) {
        blocks.push_back(&block);
        for (Operation &nested : block)
          pending.push_back(&nested);
      }
    }
  }

  // Break SSA and block-argument use lists before unlinking any operation.
  // Operation::dropAllReferences() cannot be used here because it recursively
  // visits nested regions, which is precisely the failure mode this API
  // avoids.
  for (Operation *current : operations)
    current->dropAllUses();
  for (Block *block : blocks)
    for (BlockArgument argument : block->getArguments())
      argument.dropAllUses();

  // Pre-order collection guarantees that reverse order destroys every nested
  // operation before its parent. Each native destructor therefore sees only
  // empty nested blocks and has bounded recursion depth.
  for (auto iterator = operations.rbegin(); iterator != operations.rend();
       ++iterator) {
    Operation *current = *iterator;
    if (current->getBlock())
      current->remove();
    current->destroy();
  }
}

namespace {
using MemoryEffectList =
    llvm::SmallVectorImpl<MemoryEffects::EffectInstance>;

static MemoryEffectList *
unwrapEffectList(MlirBeaverMemoryEffectInstancesList effects) {
  return static_cast<MemoryEffectList *>(effects.ptr);
}

static MlirBeaverMemoryEffectInstancesList
wrapEffectList(MemoryEffectList *effects) {
  return {effects};
}

static llvm::SmallVector<NamedAttribute>
collectInherentAttributes(Operation *operation) {
  llvm::SmallVector<NamedAttribute> attributes;
#ifdef BEAVER_HAS_MLIR_INHERENT_ATTRIBUTE_VISITOR
  operation->getName().walkInherentAttrs(
      operation, [&](llvm::StringRef name, Attribute &attribute) {
        if (attribute)
          attributes.emplace_back(
              StringAttr::get(operation->getContext(), name), attribute);
      });
#else
  NamedAttrList populated;
  operation->getName().populateInherentAttrs(operation, populated);
  attributes.append(populated.begin(), populated.end());
#endif
  return attributes;
}

#ifdef BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE
static std::mutex transientScopeMutex;
static std::unordered_set<MLIRContext *> transientScopeContexts;
#endif

class BeaverMemoryEffectsOpInterfaceFallbackModel
    : public MemoryEffectOpInterface::FallbackModel<
          BeaverMemoryEffectsOpInterfaceFallbackModel> {
public:
  void setCallbacks(MlirBeaverMemoryEffectsOpInterfaceCallbacks value) {
    callbacks = value;
  }

  ~BeaverMemoryEffectsOpInterfaceFallbackModel() {
    if (callbacks.destruct)
      callbacks.destruct(callbacks.userData);
  }

  static TypeID getInterfaceID() {
    return MemoryEffectOpInterface::getInterfaceID();
  }

  static bool classof(const MemoryEffectOpInterface::Concept *) { return true; }

  void getEffects(Operation *operation, MemoryEffectList &effects) const {
    assert(callbacks.getEffects && "getEffects callback not set");
    callbacks.getEffects(wrap(operation), wrapEffectList(&effects),
                         callbacks.userData);
  }

private:
  MlirBeaverMemoryEffectsOpInterfaceCallbacks callbacks{};
};

/// A reusable pool with a nominal MLIR parallelism limit and elastic workers.
/// MLIR uses `getMaxConcurrency()` to bound ordinary parallel algorithms. The
/// pool itself may grow past that number when active work synchronously waits
/// for BEAM callbacks which start nested MLIR work on the same shared pool.
class BeaverElasticThreadPool final : public llvm::ThreadPoolInterface {
public:
  explicit BeaverElasticThreadPool(unsigned maxConcurrency)
      : maxConcurrency(std::max(1u, maxConcurrency)) {}

  ~BeaverElasticThreadPool() override {
    {
      std::unique_lock<std::mutex> lock(mutex);
      completion.wait(lock, [this]() { return outstanding == 0; });
      shuttingDown = true;
    }
    available.notify_all();
    for (std::thread &thread : threads)
      thread.join();
  }

  void wait() override {
    std::unique_lock<std::mutex> lock(mutex);
    completion.wait(lock, [this]() { return outstanding == 0; });
  }

  void wait(llvm::ThreadPoolTaskGroup &group) override {
    std::unique_lock<std::mutex> lock(mutex);
    completion.wait(lock, [this, &group]() {
      auto it = groupOutstanding.find(&group);
      return it == groupOutstanding.end() || it->second == 0;
    });
  }

  unsigned getMaxConcurrency() const override { return maxConcurrency; }

private:
  void asyncEnqueue(llvm::unique_function<void()> task,
                    llvm::ThreadPoolTaskGroup *group) override {
    std::lock_guard<std::mutex> lock(mutex);
    assert(!shuttingDown && "queueing work during pool destruction");
    tasks.emplace_back(std::move(task), group);
    ++outstanding;
    if (group)
      ++groupOutstanding[group];

    const size_t inactiveWorkers = threads.size() - activeWorkers;
    if (tasks.size() > inactiveWorkers)
      threads.emplace_back([this]() { workerLoop(); });
    else
      available.notify_one();
  }

  void workerLoop() {
    while (true) {
      llvm::unique_function<void()> task;
      llvm::ThreadPoolTaskGroup *group = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex);
        available.wait(lock,
                       [this]() { return shuttingDown || !tasks.empty(); });
        if (shuttingDown && tasks.empty())
          return;
        task = std::move(tasks.front().first);
        group = tasks.front().second;
        tasks.pop_front();
        ++activeWorkers;
      }

      task();

      {
        std::lock_guard<std::mutex> lock(mutex);
        --activeWorkers;
        --outstanding;
        if (group) {
          auto it = groupOutstanding.find(group);
          if (--it->second == 0)
            groupOutstanding.erase(it);
        }
      }
      completion.notify_all();
    }
  }

  const unsigned maxConcurrency;
  std::mutex mutex;
  std::condition_variable available;
  std::condition_variable completion;
  std::deque<std::pair<llvm::unique_function<void()>,
                       llvm::ThreadPoolTaskGroup *>>
      tasks;
  std::vector<std::thread> threads;
  std::unordered_map<llvm::ThreadPoolTaskGroup *, size_t> groupOutstanding;
  size_t activeWorkers = 0;
  size_t outstanding = 0;
  bool shuttingDown = false;
};
} // namespace

MLIR_CAPI_EXPORTED MlirStringRef
beaverSymbolTableGetDefaultVisibilityAttributeName(void) {
#ifdef BEAVER_HAS_MLIR_DEFAULT_SYMBOL_VISIBILITY_ATTRIBUTE_NAME
  return wrap(SymbolOpInterface::getDefaultVisibilityAttrName());
#else
  return wrap(SymbolTable::getVisibilityAttrName());
#endif
}

MLIR_CAPI_EXPORTED MlirBeaverSymbolVisibility
beaverSymbolTableGetSymbolVisibility(MlirOperation symbol) {
  switch (SymbolTable::getSymbolVisibility(unwrap(symbol))) {
  case SymbolTable::Visibility::Public:
    return MlirBeaverSymbolVisibilityPublic;
  case SymbolTable::Visibility::Private:
    return MlirBeaverSymbolVisibilityPrivate;
  case SymbolTable::Visibility::Nested:
    return MlirBeaverSymbolVisibilityNested;
  }
  llvm_unreachable("unknown symbol visibility");
}

MLIR_CAPI_EXPORTED void beaverSymbolTableSetSymbolVisibility(
    MlirOperation symbol, MlirBeaverSymbolVisibility visibility) {
  if (visibility < MlirBeaverSymbolVisibilityPublic ||
      visibility > MlirBeaverSymbolVisibilityNested)
    llvm_unreachable("unknown Beaver symbol visibility");

  SymbolTable::Visibility nativeVisibility = SymbolTable::Visibility::Public;
  switch (visibility) {
  case MlirBeaverSymbolVisibilityPublic:
    nativeVisibility = SymbolTable::Visibility::Public;
    break;
  case MlirBeaverSymbolVisibilityPrivate:
    nativeVisibility = SymbolTable::Visibility::Private;
    break;
  case MlirBeaverSymbolVisibilityNested:
    nativeVisibility = SymbolTable::Visibility::Nested;
    break;
  }
  SymbolTable::setSymbolVisibility(unwrap(symbol), nativeVisibility);
}

MLIR_CAPI_EXPORTED bool beaverOperationHasInherentAttributeByName(
    MlirOperation operation, MlirStringRef name) {
  Operation *op = unwrap(operation);
  if (op->getName().getOpPropertyByteSize() == 0)
    return false;
  return op->getInherentAttr(unwrap(name)).has_value();
}

MLIR_CAPI_EXPORTED MlirAttribute beaverOperationGetInherentAttributeByName(
    MlirOperation operation, MlirStringRef name) {
  Operation *op = unwrap(operation);
  if (op->getName().getOpPropertyByteSize() == 0)
    return {};
  std::optional<Attribute> attribute = op->getInherentAttr(unwrap(name));
  return attribute ? wrap(*attribute) : MlirAttribute{};
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationGetNumInherentAttributes(MlirOperation operation) {
  return static_cast<intptr_t>(
      collectInherentAttributes(unwrap(operation)).size());
}

MLIR_CAPI_EXPORTED MlirNamedAttribute
beaverOperationGetInherentAttribute(MlirOperation operation,
                                    intptr_t position) {
  llvm::SmallVector<NamedAttribute> attributes =
      collectInherentAttributes(unwrap(operation));
  assert(position >= 0 && static_cast<size_t>(position) < attributes.size() &&
         "inherent attribute position out of bounds");
  NamedAttribute attribute = attributes[position];
  return MlirNamedAttribute{wrap(attribute.getName()),
                            wrap(attribute.getValue())};
}

MLIR_CAPI_EXPORTED void beaverMemoryEffectsOpInterfaceAttachFallbackModel(
    MlirContext context, MlirStringRef operationName,
    MlirBeaverMemoryEffectsOpInterfaceCallbacks callbacks) {
  std::optional<RegisteredOperationName> operation =
      RegisteredOperationName::lookup(unwrap(operationName), unwrap(context));
  assert(operation.has_value() && "operation not found in context");

  operation
      ->attachInterface<BeaverMemoryEffectsOpInterfaceFallbackModel>();
  auto *model = cast<BeaverMemoryEffectsOpInterfaceFallbackModel>(
      operation->getInterface<BeaverMemoryEffectsOpInterfaceFallbackModel>());
  assert(model && "failed to get Beaver MemoryEffects fallback model");
  model->setCallbacks(callbacks);
}

MLIR_CAPI_EXPORTED void beaverMemoryEffectInstancesListAppend(
    MlirBeaverMemoryEffectInstancesList effects,
    MlirMemoryEffectInstance instance) {
  unwrapEffectList(effects)->push_back(*unwrap(instance));
}

MLIR_CAPI_EXPORTED bool beaverMemoryEffectsOpInterfaceGetEffects(
    MlirOperation operation, MlirBeaverMemoryEffectInstancesCallback callback,
    void *userData) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(unwrap(operation));
  if (!interface)
    return false;

  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  llvm::SmallVector<MlirMemoryEffectInstance> wrappedEffects;
  wrappedEffects.reserve(effects.size());
  for (MemoryEffects::EffectInstance &effect : effects)
    wrappedEffects.push_back(wrap(&effect));
  callback(wrappedEffects.size(), wrappedEffects.data(), userData);
  return true;
}

MLIR_CAPI_EXPORTED MlirBeaverMemoryEffectKind
beaverMemoryEffectInstanceGetKind(MlirMemoryEffectInstance instance) {
  MemoryEffects::Effect *effect = unwrap(instance)->getEffect();
  if (effect == MemoryEffects::Allocate::get())
    return MlirBeaverMemoryEffectAllocate;
  if (effect == MemoryEffects::Free::get())
    return MlirBeaverMemoryEffectFree;
  if (effect == MemoryEffects::Read::get())
    return MlirBeaverMemoryEffectRead;
  if (effect == MemoryEffects::Write::get())
    return MlirBeaverMemoryEffectWrite;
  return MlirBeaverMemoryEffectUnknown;
}

MLIR_CAPI_EXPORTED MlirSideEffectResource
beaverMemoryEffectInstanceGetResource(MlirMemoryEffectInstance instance) {
  return wrap(unwrap(instance)->getResource());
}

MLIR_CAPI_EXPORTED int
beaverMemoryEffectInstanceGetStage(MlirMemoryEffectInstance instance) {
  return unwrap(instance)->getStage();
}

MLIR_CAPI_EXPORTED bool beaverMemoryEffectInstanceGetEffectOnFullRegion(
    MlirMemoryEffectInstance instance) {
  return unwrap(instance)->getEffectOnFullRegion();
}

MLIR_CAPI_EXPORTED MlirAttribute
beaverMemoryEffectInstanceGetParameters(MlirMemoryEffectInstance instance) {
  return wrap(unwrap(instance)->getParameters());
}

MLIR_CAPI_EXPORTED MlirOpOperand
beaverMemoryEffectInstanceGetOpOperand(MlirMemoryEffectInstance instance) {
  return wrap(unwrap(instance)->getEffectValue<OpOperand *>());
}

MLIR_CAPI_EXPORTED MlirValue
beaverMemoryEffectInstanceGetValue(MlirMemoryEffectInstance instance) {
  return wrap(unwrap(instance)->getValue());
}

MLIR_CAPI_EXPORTED MlirAttribute
beaverMemoryEffectInstanceGetSymbolRef(MlirMemoryEffectInstance instance) {
  return wrap(unwrap(instance)->getSymbolRef());
}

MLIR_CAPI_EXPORTED void beaverTransformOnlyReadsHandle(
    MlirOpOperand *operands, intptr_t numOperands,
    MlirBeaverMemoryEffectInstancesList effects) {
  MutableArrayRef<OpOperand> unwrapped;
  if (numOperands != 0)
    unwrapped = MutableArrayRef<OpOperand>(unwrap(*operands), numOperands);
  transform::onlyReadsHandle(unwrapped, *unwrapEffectList(effects));
}

MLIR_CAPI_EXPORTED void beaverTransformConsumesHandle(
    MlirOpOperand *operands, intptr_t numOperands,
    MlirBeaverMemoryEffectInstancesList effects) {
  MutableArrayRef<OpOperand> unwrapped;
  if (numOperands != 0)
    unwrapped = MutableArrayRef<OpOperand>(unwrap(*operands), numOperands);
  transform::consumesHandle(unwrapped, *unwrapEffectList(effects));
}

MLIR_CAPI_EXPORTED void beaverTransformProducesHandle(
    MlirValue *results, intptr_t numResults,
    MlirBeaverMemoryEffectInstancesList effects) {
  for (intptr_t index = 0; index < numResults; ++index) {
    auto result = cast<OpResult>(unwrap(results[index]));
    transform::producesHandle(ResultRange(result), *unwrapEffectList(effects));
  }
}

MLIR_CAPI_EXPORTED void beaverTransformModifiesPayload(
    MlirBeaverMemoryEffectInstancesList effects) {
  transform::modifiesPayload(*unwrapEffectList(effects));
}

MLIR_CAPI_EXPORTED void beaverTransformOnlyReadsPayload(
    MlirBeaverMemoryEffectInstancesList effects) {
  transform::onlyReadsPayload(*unwrapEffectList(effects));
}

MLIR_CAPI_EXPORTED bool beaverTransformPackedParamsSupported(void) {
#ifdef BEAVER_HAS_PACKED_TRANSFORM_PARAMS
  return true;
#else
  return false;
#endif
}

MLIR_CAPI_EXPORTED MlirLlvmThreadPool
beaverLlvmThreadPoolCreateElastic(unsigned maxConcurrency) {
  return wrap(static_cast<llvm::ThreadPoolInterface *>(
      new BeaverElasticThreadPool(maxConcurrency)));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverGetLLVMVersion() {
  static const std::string versionAndRevision =
      std::string(LLVM_VERSION_STRING) + "@" + LLVM_REVISION;
  return wrap(llvm::StringRef(versionAndRevision));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetArgument(MlirPass pass) {
  auto argument = unwrap(pass)->getArgument();
  return wrap(argument);
}

MLIR_CAPI_EXPORTED MlirOpPassManager beaverOpPassManagerCreate(void) {
  return wrap(new OpPassManager());
}

MLIR_CAPI_EXPORTED void
beaverOpPassManagerDestroy(MlirOpPassManager passManager) {
  delete unwrap(passManager);
}

MLIR_CAPI_EXPORTED bool
beaverCompositeFixedPointFailureActionSupported(void) {
#ifdef BEAVER_HAS_MLIR_COMPOSITE_FAILURE_ACTION
  return true;
#else
  return false;
#endif
}

MLIR_CAPI_EXPORTED MlirPass beaverCreateCompositeFixedPointPass(
    MlirStringRef name, MlirOpPassManager innerPipeline,
    intptr_t maxIterations, MlirBeaverConvergenceFailureAction action) {
  OpPassManager *source = unwrap(innerPipeline);
  auto populate = [source](OpPassManager &target) {
    target = std::move(*source);
  };

#ifdef BEAVER_HAS_MLIR_COMPOSITE_FAILURE_ACTION
  ConvergenceFailureAction nativeAction;
  switch (action) {
  case MlirBeaverConvergenceFailureWarn:
    nativeAction = ConvergenceFailureAction::Warn;
    break;
  case MlirBeaverConvergenceFailureError:
    nativeAction = ConvergenceFailureAction::Error;
    break;
  case MlirBeaverConvergenceFailureSilent:
    nativeAction = ConvergenceFailureAction::Silent;
    break;
  }
  return wrap(createCompositeFixedPointPass(unwrap(name).str(), populate,
                                            maxIterations, nativeAction)
                  .release());
#else
  assert(action == MlirBeaverConvergenceFailureWarn &&
         "linked LLVM cannot configure composite convergence failure");
  return wrap(createCompositeFixedPointPass(unwrap(name).str(), populate,
                                            maxIterations)
                  .release());
#endif
}

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetName(MlirPass pass) {
  auto argument = unwrap(pass)->getName();
  return wrap(argument);
}

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetDescription(MlirPass pass) {
  return wrap(unwrap(pass)->getDescription());
}

MLIR_CAPI_EXPORTED MlirContext
beaverPassManagerGetContext(MlirPassManager passManager) {
  return wrap(unwrap(passManager)->getContext());
}

MLIR_CAPI_EXPORTED void
beaverPassManagerEnableTiming(MlirPassManager passManager) {
  unwrap(passManager)->enableTiming();
}

MLIR_CAPI_EXPORTED void beaverContextGetOps(MlirContext context,
                                            MlirStringCallback insert,
                                            void *container) {
  for (const RegisteredOperationName &op :
       unwrap(context)->getRegisteredOperations()) {
    insert(wrap(op.getStringRef()), container);
  }
}

MLIR_CAPI_EXPORTED void beaverContextGetDialects(MlirContext context,
                                                 MlirStringCallback insert,
                                                 void *container) {
  for (auto dialect : unwrap(context)->getAvailableDialects()) {
    insert(wrap(dialect), container);
  }
}

MLIR_CAPI_EXPORTED const char *
beaverStringRefGetData(MlirStringRef string_ref) {
  return string_ref.data;
}

MLIR_CAPI_EXPORTED size_t beaverStringRefGetLength(MlirStringRef string_ref) {
  return string_ref.length;
}

MLIR_CAPI_EXPORTED uint64_t
beaverOperationStructuralHashValue(MlirOperation op, uint32_t flags) {
  return static_cast<uint64_t>(mlirOperationStructuralHashValue(op, flags));
}

MLIR_CAPI_EXPORTED void beaverIRMappingClear(MlirIRMapping mapping) {
  IRMapping *cppMapping = unwrap(mapping);
  SmallVector<Block *> blocks;
  SmallVector<Operation *> operations;

  for (auto [from, _] : cppMapping->getBlockMap())
    blocks.push_back(from);
  for (auto [from, _] : cppMapping->getOperationMap())
    operations.push_back(from);

  cppMapping->clear();
  for (Block *block : blocks)
    cppMapping->erase(block);
  for (Operation *operation : operations)
    cppMapping->erase(operation);
}

MLIR_CAPI_EXPORTED bool beaverIsNullContext(MlirContext w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullDialect(MlirDialect w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullDialectRegistry(MlirDialectRegistry w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullLocation(MlirLocation w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullModule(MlirModule w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullOperation(MlirOperation w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullRegion(MlirRegion w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullBlock(MlirBlock w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullValue(MlirValue w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullType(MlirType w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullAttribute(MlirAttribute w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullSymbolTable(MlirSymbolTable w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullExecutionEngine(MlirExecutionEngine w) {
  return !w.ptr;
}

MLIR_CAPI_EXPORTED MlirLocation
beaverOperationStateGetLocation(MlirOperationState state) {
  return state.location;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumResults(MlirOperationState state) {
  return state.nResults;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumOperands(MlirOperationState state) {
  return state.nOperands;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumRegions(MlirOperationState state) {
  return state.nRegions;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumAttributes(MlirOperationState state) {
  return state.nAttributes;
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverOperationStateGetName(MlirOperationState state) {
  return state.name;
}

MLIR_CAPI_EXPORTED MlirOperation
beaverOperationCreate(MlirOperationState state) {
  return mlirOperationCreate(&state);
}

MLIR_CAPI_EXPORTED MlirContext
beaverOperationStateGetContext(MlirOperationState state) {
  return mlirLocationGetContext(state.location);
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultSuccess() {
  return mlirLogicalResultSuccess();
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultFailure() {
  return mlirLogicalResultFailure();
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsSuccess(MlirLogicalResult res) {
  return mlirLogicalResultIsSuccess(res);
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsFailure(MlirLogicalResult res) {
  return mlirLogicalResultIsFailure(res);
}

MLIR_CAPI_EXPORTED
MlirIdentifier beaverNamedAttributeGetName(MlirNamedAttribute na) {
  return na.name;
}

MLIR_CAPI_EXPORTED
MlirAttribute beaverNamedAttributeGetAttribute(MlirNamedAttribute na) {
  return na.attribute;
}

MLIR_CAPI_EXPORTED MlirPass beaverPassCreate(
    void (*construct)(void *userData), void (*destruct)(void *userData),
    MlirLogicalResult (*initialize)(MlirContext ctx, void *userData),
    void *(*clone)(void *userData),
    void (*run)(MlirOperation op, MlirExternalPass pass, void *userData),
    MlirTypeID passID, MlirStringRef name, MlirStringRef argument,
    MlirStringRef description, MlirStringRef opName,
    intptr_t nDependentDialects, MlirDialectHandle *dependentDialects,
    void *userData) {
  return mlirCreateExternalPass(
      passID, name, argument, description, opName, nDependentDialects,
      dependentDialects,
      MlirExternalPassCallbacks{construct, destruct, initialize, clone, run},
      userData);
}

MLIR_CAPI_EXPORTED void beaverLocationPrint(MlirLocation location,
                                            MlirStringCallback callback,
                                            void *userData) {
  if (auto loc = mlir::dyn_cast<FileLineColLoc>(unwrap(location))) {
    std::string s = loc.getFilename().str() + ":" +
                    std::to_string(loc.getLine()) + ":" +
                    std::to_string(loc.getColumn());
    callback(wrap(s), userData);
  } else {
    mlirLocationPrint(location, callback, userData);
  }
}

MLIR_CAPI_EXPORTED MlirLocation
beaverLocationFusedGetLocationAt(MlirLocation location, intptr_t position) {
  FusedLoc fused = llvm::cast<FusedLoc>(unwrap(location));
  ArrayRef<Location> locations = fused.getLocations();
  assert(position >= 0 && static_cast<size_t>(position) < locations.size() &&
         "fused location index out of bounds");
  return wrap(locations[position]);
}

MLIR_CAPI_EXPORTED void mlirIdentifierPrint(MlirIdentifier identifier,
                                            MlirStringCallback callback,
                                            void *userData) {
  callback(mlirIdentifierStr(identifier), userData);
}

MLIR_CAPI_EXPORTED void beaverOperationPrintSpecializedFrom(
    MlirOperation op, MlirStringCallback callback, void *userData) {
  mlirOperationPrintWithFlags(
      op, wrap(&OpPrintingFlags().useLocalScope().printGenericOpForm(false)),
      callback, userData);
}

MLIR_CAPI_EXPORTED void
beaverOperationPrintGenericOpForm(MlirOperation op, MlirStringCallback callback,
                                  void *userData) {
  mlirOperationPrintWithFlags(
      op, wrap(&OpPrintingFlags().useLocalScope().printGenericOpForm(true)),
      callback, userData);
}

MLIR_CAPI_EXPORTED void beaverOperationDumpGeneric(MlirOperation op) {
  unwrap(op)->print(llvm::errs(),
                    OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs() << "\n";
}

MLIR_CAPI_EXPORTED MlirGreedyRewriteDriverConfig
beaverGreedyRewriteDriverConfigGet() {
  return mlirGreedyRewriteDriverConfigCreate();
}

MLIR_CAPI_EXPORTED bool beaverContextAddWork(MlirContext context,
                                             void (*task)(void *), void *arg) {
  if (!unwrap(context)->isMultithreadingEnabled())
    return false;

  // Callback bridges may recursively schedule more bridge work (for example,
  // an Elixir pass applying an Elixir rewrite pattern). The application-owned
  // elastic pool can grow when all current workers wait for nested callbacks.
  unwrap(mlirContextGetThreadPool(context))->async(
      [task, arg]() { task(arg); });
  return true;
}

MLIR_CAPI_EXPORTED bool beaverContextTransientScopeSupported(void) {
#ifdef BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE
  return true;
#else
  return false;
#endif
}

MLIR_CAPI_EXPORTED bool
beaverContextBeginTransientScope(MlirContext context) {
#ifdef BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE
  MLIRContext *unwrapped = unwrap(context);
  std::lock_guard<std::mutex> lock(transientScopeMutex);
  if (!transientScopeContexts.insert(unwrapped).second)
    return false;
  unwrapped->beginTransientScope();
  return true;
#else
  (void)context;
  return false;
#endif
}

MLIR_CAPI_EXPORTED bool beaverContextEndTransientScope(MlirContext context) {
#ifdef BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE
  MLIRContext *unwrapped = unwrap(context);
  std::lock_guard<std::mutex> lock(transientScopeMutex);
  auto active = transientScopeContexts.find(unwrapped);
  if (active == transientScopeContexts.end())
    return false;
  unwrapped->endTransientScope();
  transientScopeContexts.erase(active);
  return true;
#else
  (void)context;
  return false;
#endif
}

MLIR_CAPI_EXPORTED bool
beaverContextHasActiveTransientScope(MlirContext context) {
#ifdef BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE
  std::lock_guard<std::mutex> lock(transientScopeMutex);
  return transientScopeContexts.find(unwrap(context)) !=
         transientScopeContexts.end();
#else
  (void)context;
  return false;
#endif
}

MLIR_CAPI_EXPORTED MlirType beaverDenseElementsAttrGetType(MlirAttribute attr) {
  return wrap(llvm::cast<DenseElementsAttr>(unwrap(attr)).getType());
}

MLIR_CAPI_EXPORTED intptr_t beaverShapedTypeGetNumElements(MlirType type) {
  return llvm::cast<ShapedType>(unwrap(type)).getNumElements();
}
