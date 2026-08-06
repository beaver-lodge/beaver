#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/IRMapping.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir-c/Beaver/ActionTracing.h"
#include "mlir/Debug/BreakpointManagers/TagBreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Unit.h"
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
#include <vector>

using namespace mlir;

namespace {
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

MLIR_CAPI_EXPORTED bool beaverIsOpNameTerminator(MlirStringRef op_name,
                                                 MlirContext context) {
  auto name = OperationName(unwrap(op_name), unwrap(context));
  return name.isRegistered() && name.mightHaveTrait<OpTrait::IsTerminator>();
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

MLIR_CAPI_EXPORTED MlirContext
beaverOperationStateGetContext(MlirOperationState state) {
  return mlirLocationGetContext(state.location);
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

MLIR_CAPI_EXPORTED MlirAttribute beaverGetReassociationIndicesForReshape(
    MlirType sourceType, MlirType targetType) {
  auto indices = mlir::getReassociationIndicesForReshape(
      mlir::cast<RankedTensorType>(unwrap(sourceType)),
      mlir::cast<RankedTensorType>(unwrap(targetType)));
  OpBuilder b{unwrap(sourceType).getContext()};
  if (!indices) {
    return wrap(Attribute{});
  }
  return wrap(getReassociationIndicesAttribute(b, *indices));
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

template <typename T, typename EntityLookup, typename EntityGetter>
T getIRDLDefinedEntity(MlirStringRef dialect, MlirStringRef name,
                       MlirAttribute attrArr, EntityLookup lookup,
                       EntityGetter getter) {
  if (auto d =
          unwrap(attrArr).getContext()->getOrLoadDialect(unwrap(dialect))) {
    if (auto e = mlir::dyn_cast<ExtensibleDialect>(d)) {
      if (auto definition = lookup(e, unwrap(name))) {
        if (auto arr = mlir::dyn_cast<ArrayAttr>(unwrap(attrArr))) {
          return getter(definition, arr.getValue());
        }
      }
    }
  }
  return {};
}

MLIR_CAPI_EXPORTED MlirType beaverIRDLGetDefinedType(MlirStringRef dialect,
                                                     MlirStringRef type,
                                                     MlirAttribute params) {

  return wrap(getIRDLDefinedEntity<Type>(
      dialect, type, params,
      [](auto d, auto name) { return d->lookupTypeDefinition(name); },
      DynamicType::get));
}

MLIR_CAPI_EXPORTED MlirAttribute beaverIRDLGetDefinedAttr(
    MlirStringRef dialect, MlirStringRef attr, MlirAttribute params) {

  return wrap(getIRDLDefinedEntity<Attribute>(
      dialect, attr, params,
      [](auto d, auto name) { return d->lookupAttrDefinition(name); },
      DynamicAttr::get));
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

MLIR_CAPI_EXPORTED MlirType beaverDenseElementsAttrGetType(MlirAttribute attr) {
  return wrap(llvm::cast<DenseElementsAttr>(unwrap(attr)).getType());
}

MLIR_CAPI_EXPORTED intptr_t beaverShapedTypeGetNumElements(MlirType type) {
  return llvm::cast<ShapedType>(unwrap(type)).getNumElements();
}

#include "mlir-c/Debug.h"
MLIR_CAPI_EXPORTED void beaverSetGlobalDebugTypes(const MlirStringRef *types,
                                                  intptr_t n) {
  // Convert MlirStringRef array to array of C strings
  std::vector<const char *> cstrings;
  cstrings.reserve(n);
  for (intptr_t i = 0; i < n; ++i) {
    cstrings.push_back(beaverStringRefGetData(types[i]));
  }

  // Call the underlying MLIR function
  mlirSetGlobalDebugTypes(cstrings.data(), n);
}

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

MLIR_CAPI_EXPORTED MlirStringRef beaverGetNumWorkgroupAttributionsAttrName() {
  return wrap(llvm::StringRef("workgroup_attributions"));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverGetContainerModuleAttrName() {
  return wrap(gpu::GPUDialect::getContainerModuleAttrName());
}

//===----------------------------------------------------------------------===//
// Action Tracing
//===----------------------------------------------------------------------===//

using namespace mlir::tracing;

namespace {

/// Render a location as a compact string without full IR text.
static std::string printLocation(Location loc) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  loc.print(os);
  os.flush();
  return buffer;
}

/// Serialize one IR unit into a compact JSON object: {"kind": ..., "name": ...,
/// "loc": ...}. Never materializes full IR text.
static void serializeIRUnit(const IRUnit &unit, llvm::raw_string_ostream &os) {
  os << "{\"kind\":";
  if (isa<Operation *>(unit)) {
    auto *op = cast<Operation *>(unit);
    os << "\"operation\",\"name\":\"";
    os.write_escaped(op->getName().getStringRef());
    os << "\",\"loc\":\"";
    os.write_escaped(printLocation(op->getLoc()));
    os << "\"}";
    return;
  }
  if (isa<Region *>(unit)) {
    os << "\"region\"}";
    return;
  }
  if (isa<Block *>(unit)) {
    os << "\"block\"}";
    return;
  }
  if (isa<Value>(unit)) {
    Value value = cast<Value>(unit);
    os << "\"value\",\"loc\":\"";
    os.write_escaped(printLocation(value.getLoc()));
    os << "\"}";
    return;
  }
  os << "\"unknown\"}";
}

/// Serialize the IR units associated with an action.
static std::string serializeIRUnits(ArrayRef<IRUnit> units) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "[";
  bool first = true;
  for (const IRUnit &unit : units) {
    if (!first)
      os << ",";
    first = false;
    serializeIRUnit(unit, os);
  }
  os << "]";
  os.flush();
  return buffer;
}

class ActionTracingSession {
public:
  ActionTracingSession(MLIRContext *context, std::string filterJson,
                       std::string locationJson, std::string skipJson,
                       std::string limitJson)
      : context(context), filterTags(parseTagList(filterJson)),
        filterLocations(parseStringList(locationJson)),
        skipCounts(parseCountMap(skipJson)), limits(parseCountMap(limitJson)),
        controlCallback([this](const ActionActiveStack *stack) {
          return controlAction(stack);
        }),
        executionContext(controlCallback) {
    // Breakpoints make the control callback run: ExecutionContext only invokes
    // the callback when a breakpoint matches the action. Tags named in
    // :skip/:limit therefore get a breakpoint so their first occurrences can be
    // skipped or limited.
    for (const auto &entry : skipCounts)
      tagBreakpoints.addBreakpoint(entry.first);
    for (const auto &entry : limits)
      tagBreakpoints.addBreakpoint(entry.first);
    executionContext.addBreakpointManager(&tagBreakpoints);
  }

  ~ActionTracingSession() {
    context->registerActionHandler(nullptr);
    // Drop any pending events; the session is being torn down.
    std::lock_guard<std::mutex> lock(mutex);
    events.clear();
  }

  void attach() {
    context->registerActionHandler(
        [this](function_ref<void()> transform, const Action &action) {
          executionContext(transform, action);
        });
    executionContext.registerObserver(&observer);
  }

  bool drain(MlirBeaverActionEventsCallback callback, void *userData) {
    std::vector<std::string> drained;
    {
      std::lock_guard<std::mutex> lock(mutex);
      drained.swap(events);
    }
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    os << "[";
    bool first = true;
    for (const std::string &event : drained) {
      if (!first)
        os << ",";
      first = false;
      os << event;
    }
    os << "]";
    os.flush();
    callback(buffer.c_str(), userData);
    return true;
  }

  MLIRContext *getContext() const { return context; }

private:
  struct Observer final : public ExecutionContext::Observer {
    Observer(ActionTracingSession *session) : session(session) {}

    ActionTracingSession *session;

    void beforeExecute(const ActionActiveStack *stack, Breakpoint *,
                       bool willExecute) override {
      if (!willExecute)
        return;
      const Action &action = stack->getAction();
      if (!session->shouldObserve(action))
        return;
      std::string description;
      llvm::raw_string_ostream os(description);
      action.print(os);
      os.flush();
      std::string event = "{\"phase\":\"before\",\"tag\":\"";
      event += escapeJson(action.getTag());
      event += "\",\"depth\":";
      event += std::to_string(stack->getDepth());
      event += ",\"description\":\"";
      event += escapeJson(description);
      event += "\",\"ir_units\":";
      event += serializeIRUnits(action.getContextIRUnits());
      event += ",\"t_ns\":";
      event += std::to_string(nowNs());
      event += "}";
      session->push(event);
    }

    void afterExecute(const ActionActiveStack *stack) override {
      const Action &action = stack->getAction();
      if (!session->shouldObserve(action))
        return;
      std::string event = "{\"phase\":\"after\",\"tag\":\"";
      event += escapeJson(action.getTag());
      event += "\",\"depth\":";
      event += std::to_string(stack->getDepth());
      event += ",\"t_ns\":";
      event += std::to_string(nowNs());
      event += "}";
      session->push(event);
    }
  };

  ExecutionContext::Control controlAction(const ActionActiveStack *stack) {
    const Action &action = stack->getAction();
    std::string tag = action.getTag().str();
    std::lock_guard<std::mutex> lock(mutex);

    auto skipIt = skipCounts.find(tag);
    if (skipIt != skipCounts.end()) {
      if (skipIt->second > 0) {
        skipIt->second--;
        return ExecutionContext::Skip;
      }
      skipCounts.erase(skipIt);
    }

    auto limitIt = limits.find(tag);
    if (limitIt != limits.end()) {
      if (limitIt->second == 0)
        return ExecutionContext::Skip;
      limitIt->second--;
    }

    return ExecutionContext::Apply;
  }

  bool shouldObserve(const Action &action) const {
    if (!filterTags.empty()) {
      std::string tag = action.getTag().str();
      if (std::find(filterTags.begin(), filterTags.end(), tag) ==
          filterTags.end())
        return false;
    }
    if (!filterLocations.empty()) {
      bool found = false;
      for (const IRUnit &unit : action.getContextIRUnits()) {
        std::optional<Location> loc;
        if (isa<Operation *>(unit))
          loc = cast<Operation *>(unit)->getLoc();
        else if (isa<Value>(unit))
          loc = cast<Value>(unit).getLoc();
        if (!loc)
          continue;
        std::string printed = printLocation(*loc);
        for (const std::string &needle : filterLocations) {
          if (printed.find(needle) != std::string::npos) {
            found = true;
            break;
          }
        }
        if (found)
          break;
      }
      if (!found)
        return false;
    }
    return true;
  }

  void push(std::string event) {
    std::lock_guard<std::mutex> lock(mutex);
    events.push_back(std::move(event));
  }

  static std::vector<std::string> parseTagList(llvm::StringRef json) {
    return parseStringList(json);
  }

  static std::vector<std::string> parseStringList(llvm::StringRef json) {
    if (json.empty())
      return {};
    std::vector<std::string> tags;
    // Extremely small JSON array parser: ["a","b"]
    llvm::StringRef rest = json.trim();
    rest = rest.drop_front().drop_back(); // strip [ ]
    while (!rest.empty()) {
      rest = rest.trim().ltrim(',');
      if (rest.empty())
        break;
      rest = rest.trim();
      if (!rest.consume_front("\""))
        break;
      size_t end = rest.find('"');
      if (end == llvm::StringRef::npos)
        break;
      tags.push_back(rest.substr(0, end).str());
      rest = rest.substr(end + 1);
    }
    return tags;
  }

  static std::unordered_map<std::string, uint64_t>
  parseCountMap(llvm::StringRef json) {
    std::unordered_map<std::string, uint64_t> result;
    if (json.empty())
      return result;
    llvm::StringRef rest = json.trim().drop_front().drop_back(); // strip { }
    while (!rest.empty()) {
      rest = rest.trim().ltrim(',');
      if (rest.empty())
        break;
      rest = rest.trim();
      if (!rest.consume_front("\""))
        break;
      size_t end = rest.find('"');
      if (end == llvm::StringRef::npos)
        break;
      std::string key = rest.substr(0, end).str();
      rest = rest.substr(end + 1).trim();
      if (!rest.consume_front(":"))
        break;
      rest = rest.trim();
      size_t numEnd = rest.find_first_of(",}");
      uint64_t value = 0;
      rest.substr(0, numEnd).getAsInteger(10, value);
      result[std::move(key)] = value;
      rest = rest.substr(numEnd == llvm::StringRef::npos ? rest.size() : numEnd);
    }
    return result;
  }

  static std::string escapeJson(llvm::StringRef value) {
    std::string out;
    llvm::raw_string_ostream os(out);
    os.write_escaped(value);
    os.flush();
    return out;
  }

  static uint64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  MLIRContext *context;
  std::vector<std::string> filterTags;
  std::vector<std::string> filterLocations;
  std::unordered_map<std::string, uint64_t> skipCounts;
  std::unordered_map<std::string, uint64_t> limits;
  std::function<ExecutionContext::Control(const ActionActiveStack *)>
      controlCallback;
  tracing::TagBreakpointManager tagBreakpoints;
  ExecutionContext executionContext;
  Observer observer{this};
  std::mutex mutex;
  std::vector<std::string> events;
};

} // namespace

MLIR_CAPI_EXPORTED MlirBeaverActionTracing
mlirBeaverActionTracingAttach(MlirContext context, MlirStringRef filter_json,
                              MlirStringRef location_json,
                              MlirStringRef skip_json, MlirStringRef limit_json) {
  auto *session = new ActionTracingSession(
      unwrap(context),
      llvm::StringRef(filter_json.data, filter_json.length).str(),
      llvm::StringRef(location_json.data, location_json.length).str(),
      llvm::StringRef(skip_json.data, skip_json.length).str(),
      llvm::StringRef(limit_json.data, limit_json.length).str());
  session->attach();
  return MlirBeaverActionTracing{session};
}

MLIR_CAPI_EXPORTED bool mlirBeaverActionTracingDrain(
    MlirBeaverActionTracing tracing, MlirBeaverActionEventsCallback callback,
    void *user_data) {
  auto *session = static_cast<ActionTracingSession *>(tracing.ptr);
  if (!session)
    return false;
  return session->drain(callback, user_data);
}

MLIR_CAPI_EXPORTED void
mlirBeaverActionTracingDetach(MlirBeaverActionTracing tracing) {
  delete static_cast<ActionTracingSession *>(tracing.ptr);
}
