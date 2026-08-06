#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/IRMapping.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Rewrite.h"
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
