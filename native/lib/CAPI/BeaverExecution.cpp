#include "mlir-c/Beaver/Context.h"
#include "mlir-c/Beaver/Pass.h"
#include "mlir/CAPI/Execution.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir-c/Beaver/Op.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Clock = std::chrono::steady_clock;
static uint64_t now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
uint64_t beaverExecutionClock() { return now(); }

namespace beaver {
struct Metrics {
  std::atomic<uint64_t> callbacks{0}, callbackNs{0};
};
class Executor;
struct Ticket {
  Executor *executor;
  unsigned waiters = 0; // Protected by the executor mutex, including MLIR child threads.
  bool lent = false, resuming = false;
};
struct ExecutionState {
  std::shared_ptr<Ticket> ticket;
  std::shared_ptr<Metrics> metrics;
};
static thread_local Execution current;
Execution currentExecution() { return current; }
ExecutionScope::ExecutionScope(Execution execution) : previous(std::move(current)) { current = std::move(execution); }
ExecutionScope::~ExecutionScope() { current = std::move(previous); }

// Admission limits runnable root jobs. A root waiting for BEAM lends its slot
// to queued work, so nested callbacks cannot exhaust a fixed-size pool. MLIR
// children inherit the ticket and lend the root's slot at most once.
class Executor {
  struct Worker {
    std::thread thread;
    std::atomic<bool> finished{false};
  };
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::function<void()>> queue;
  std::vector<std::unique_ptr<Worker>> workers;
  unsigned capacity, idleMs, live = 0, running = 0, active = 0, resuming = 0;
  unsigned peakRunning = 0, peakThreads = 0;
  bool closing = false;

  bool available() const { return !queue.empty() && running < capacity && resuming == 0; }
  void provision() {
    if (!available()) return;
    if (live > active) {
      ready.notify_all();
      return;
    }
    auto worker = std::make_unique<Worker>();
    Worker *entry = worker.get();
    ++live;
    peakThreads = std::max(peakThreads, live);
    entry->thread = std::thread([this, entry]() { loop(entry); });
    workers.push_back(std::move(worker));
  }
  void loop(Worker *entry) {
    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
      if (!ready.wait_for(lock, std::chrono::milliseconds(idleMs), [this] { return closing || available(); })) break;
      if (closing) break;
      auto task = std::move(queue.front());
      queue.pop_front();
      ++running;
      ++active;
      peakRunning = std::max(peakRunning, running);
      provision();
      lock.unlock();
      auto state = std::make_shared<ExecutionState>();
      state->ticket = std::make_shared<Ticket>(Ticket{this});
      {
        ExecutionScope scope(state);
        task();
      }
      lock.lock();
      --running;
      --active;
      ready.notify_all();
    }
    --live;
    entry->finished = true;
  }

public:
  Executor(unsigned capacity, unsigned idleMs) : capacity(capacity), idleMs(idleMs) {}
  ~Executor() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      closing = true;
      ready.notify_all();
    }
    for (auto &worker : workers) worker->thread.join();
  }
  void enqueue(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(mutex);
    // Reap retired threads without retaining one join handle per burst.
    for (auto it = workers.begin(); it != workers.end();) {
      if ((*it)->finished) {
        (*it)->thread.join();
        it = workers.erase(it);
      } else
        ++it;
    }
    queue.push_back(std::move(task));
    provision();
  }
  void beginCallback(Ticket &ticket) {
    std::lock_guard<std::mutex> lock(mutex);
    ++ticket.waiters;
    if (!ticket.lent) {
      ticket.lent = true;
      --running;
      provision();
    }
    ready.notify_all();
  }
  void endCallback(Ticket &ticket) {
    std::unique_lock<std::mutex> lock(mutex);
    if (--ticket.waiters != 0 || ticket.resuming) {
      ready.notify_all();
      return;
    }
    ticket.resuming = true;
    ++resuming;
    ready.wait(lock, [this, &ticket] { return running < capacity || ticket.waiters != 0; });
    --resuming;
    ticket.resuming = false;
    if (ticket.waiters == 0) {
      ticket.lent = false;
      ++running;
      peakRunning = std::max(peakRunning, running);
    }
    ready.notify_all();
  }
  bool idle() {
    std::lock_guard<std::mutex> lock(mutex);
    return active == 0 && queue.empty();
  }
  bool finish() {
    std::unique_lock<std::mutex> lock(mutex);
    // A result may reach BEAM just before the worker retires its bookkeeping.
    // Do not mistake that tiny completion window for outstanding user work.
    return ready.wait_for(lock, std::chrono::milliseconds(10), [this] { return active == 0 && queue.empty(); });
  }
  void snapshot(uint64_t *out) {
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t values[] = {capacity, live, running, active - running, queue.size(), peakRunning, peakThreads};
    std::copy(std::begin(values), std::end(values), out);
  }
};
struct ContextExecution {
  std::shared_ptr<Executor> executor;
  bool callbacks = false;
};
static std::mutex contextsMutex;
static std::unordered_map<void *, ContextExecution> contexts;
static Executor &executor(ContextExecution &entry) {
  if (!entry.executor)
    entry.executor = std::make_shared<Executor>(std::max(1u, std::thread::hardware_concurrency()), 1000);
  return *entry.executor;
}
} // namespace beaver

bool beaverContextAddWork(MlirContext context, void (*task)(void *), void *arg) {
  if (!beaverContextIsThreadSafeLockingEnabled(context)) return false;
  std::lock_guard<std::mutex> lock(beaver::contextsMutex);
  beaver::executor(beaver::contexts[context.ptr]).enqueue([task, arg] { task(arg); });
  return true;
}
bool beaverExecutorConfigure(MlirContext context, unsigned capacity, unsigned idleMs) {
  if (!capacity || !idleMs) return false;
  std::shared_ptr<beaver::Executor> retired;
  {
    std::lock_guard<std::mutex> lock(beaver::contextsMutex);
    auto &entry = beaver::contexts[context.ptr];
    if (entry.executor && !entry.executor->idle()) return false;
    retired = std::move(entry.executor);
    entry.executor = std::make_shared<beaver::Executor>(capacity, idleMs);
  }
  return true;
}
bool beaverExecutorReleaseContext(MlirContext context) {
  std::shared_ptr<beaver::Executor> retired;
  {
    std::lock_guard<std::mutex> lock(beaver::contextsMutex);
    auto it = beaver::contexts.find(context.ptr);
    if (it == beaver::contexts.end()) return true;
    retired = it->second.executor;
  }
  // Neither draining bookkeeping nor joining threads holds the context registry
  // lock: an unrelated context (or a nested submission) must keep progressing.
  if (retired && !retired->finish()) return false;
  {
    std::lock_guard<std::mutex> lock(beaver::contextsMutex);
    auto it = beaver::contexts.find(context.ptr);
    if (it == beaver::contexts.end()) return true;
    if (it->second.executor != retired || (retired && !retired->idle())) return false;
    beaver::contexts.erase(it);
  }
  return true;
}
void beaverExecutorSnapshot(MlirContext context, uint64_t *out) {
  std::lock_guard<std::mutex> lock(beaver::contextsMutex);
  beaver::executor(beaver::contexts[context.ptr]).snapshot(out);
}
void beaverExecutionMarkCallbacks(MlirContext context) {
  std::lock_guard<std::mutex> lock(beaver::contextsMutex);
  beaver::contexts[context.ptr].callbacks = true;
}
bool beaverExecutionHasCallbacks(MlirContext context) {
  std::lock_guard<std::mutex> lock(beaver::contextsMutex);
  auto it = beaver::contexts.find(context.ptr);
  return it != beaver::contexts.end() && it->second.callbacks;
}
struct Measurement {
  beaver::Execution state = std::make_shared<beaver::ExecutionState>();
  std::unique_ptr<beaver::ExecutionScope> scope;
  Measurement() {
    if (auto parent = beaver::currentExecution()) state->ticket = parent->ticket;
    state->metrics = std::make_shared<beaver::Metrics>();
    scope = std::make_unique<beaver::ExecutionScope>(state);
  }
};
void *beaverExecutionMeasureBegin() { return new Measurement; }
void beaverExecutionMeasureEnd(void *opaque, uint64_t *out) {
  auto *measurement = static_cast<Measurement *>(opaque);
  out[0] = measurement->state->metrics->callbacks.load();
  out[1] = measurement->state->metrics->callbackNs.load();
  delete measurement;
}
uint64_t beaverExecutionCallbackBegin() {
  if (auto state = beaver::currentExecution()) {
    if (state->ticket) state->ticket->executor->beginCallback(*state->ticket);
    if (state->metrics) return now();
  }
  return 0;
}
void beaverExecutionCallbackEnd(uint64_t start) {
  if (auto state = beaver::currentExecution()) {
    if (start && state->metrics) {
      ++state->metrics->callbacks;
      state->metrics->callbackNs += now() - start;
    }
    if (state->ticket) state->ticket->executor->endCallback(*state->ticket);
  }
}

// LLVM-owned or caller-provided pools cannot inherit our thread-local state.
// Recover the owning execution from the callback's operation in those pools.
// Nested submissions on the same IR temporarily shadow the suspended caller.
namespace {
struct OperationBinding {
  mlir::Operation *op;
  beaver::Execution execution;
};
std::mutex bindingsMutex;
std::unordered_map<mlir::Operation *, std::vector<OperationBinding *>> bindings;
beaver::Execution operationExecution(mlir::Operation *op) {
  std::lock_guard<std::mutex> lock(bindingsMutex);
  for (; op; op = op->getParentOp()) {
    auto it = bindings.find(op);
    if (it != bindings.end()) return it->second.back()->execution;
  }
  return {};
}
} // namespace
void *beaverExecutionBindOperation(MlirOperation operation) {
  if (!unwrap(operation)->getContext()->isMultithreadingEnabled()) return nullptr;
  auto execution = beaver::currentExecution();
  if (!execution) return nullptr;
  auto *binding = new OperationBinding{unwrap(operation), std::move(execution)};
  std::lock_guard<std::mutex> lock(bindingsMutex);
  bindings[binding->op].push_back(binding);
  return binding;
}
void beaverExecutionUnbindOperation(void *opaque) {
  auto *binding = static_cast<OperationBinding *>(opaque);
  if (!binding) return;
  std::lock_guard<std::mutex> lock(bindingsMutex);
  auto &entries = bindings.at(binding->op);
  entries.erase(std::find(entries.begin(), entries.end(), binding));
  if (entries.empty()) bindings.erase(binding->op);
  delete binding;
}
void *beaverExecutionEnterOperation(MlirOperation operation) {
  if (beaver::currentExecution()) return nullptr;
  if (auto execution = operationExecution(unwrap(operation))) return new beaver::ExecutionScope(std::move(execution));
  return nullptr;
}
void beaverExecutionLeaveOperation(void *scope) { delete static_cast<beaver::ExecutionScope *>(scope); }

namespace {
std::mutex instrumentationsMutex;
std::unordered_set<mlir::PassManager *> instrumentedManagers;
thread_local std::vector<std::unique_ptr<beaver::ExecutionScope>> passScopes;

// An adaptor can clone nested passes on an LLVM-owned worker before invoking
// an external pass's run callback. Propagate ownership across the whole native
// pass, including lifecycle callbacks and verification, without retaining run
// state in a reusable pass manager.
class ExecutionInstrumentation final : public mlir::PassInstrumentation {
  mlir::PassManager *manager;

public:
  explicit ExecutionInstrumentation(mlir::PassManager *manager) : manager(manager) {}
  ~ExecutionInstrumentation() override {
    std::lock_guard<std::mutex> lock(instrumentationsMutex);
    instrumentedManagers.erase(manager);
  }
  bool isThreadSafe() const override { return true; }
  void runBeforePass(mlir::Pass *, mlir::Operation *op) override {
    auto execution = operationExecution(op);
    passScopes.push_back(execution ? std::make_unique<beaver::ExecutionScope>(std::move(execution)) : nullptr);
  }
  void runAfterPass(mlir::Pass *, mlir::Operation *) override { passScopes.pop_back(); }
  void runAfterPassFailed(mlir::Pass *, mlir::Operation *) override { passScopes.pop_back(); }
};
} // namespace

void beaverPassManagerTrackExecution(MlirPassManager passManager) {
  auto *manager = unwrap(passManager);
  if (!manager->getContext()->isMultithreadingEnabled()) return;
  std::lock_guard<std::mutex> lock(instrumentationsMutex);
  if (instrumentedManagers.insert(manager).second)
    manager->addInstrumentation(std::make_unique<ExecutionInstrumentation>(manager));
}

bool beaverOperationVerify(MlirOperation operation, bool recursive) {
  return mlir::succeeded(mlir::verify(unwrap(operation), recursive));
}

MlirModule beaverModuleCreateParseTracked(MlirContext context, MlirStringRef source) {
  // Establish ownership before the post-parse verifier fans out into an
  // LLVM-owned pool. Parsing and verification retain their ordinary semantics.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(unwrap(source), mlir::ParserConfig(unwrap(context), false));
  if (!module) return {nullptr};
  auto *binding = beaverExecutionBindOperation(wrap(module->getOperation()));
  auto valid = mlir::verify(module->getOperation());
  beaverExecutionUnbindOperation(binding);
  if (mlir::failed(valid)) return {nullptr};
  return {module.release().getOperation()};
}

// A small declarative subset with no BEAM callbacks or mutable pattern state.
// Larger/structural programs can still use PDL or compiler-kernel providers.
struct BeaverConstraint {
  int kind;
  intptr_t index, other;
  MlirStringRef name;
  MlirAttribute attribute;
  MlirType type;
};
struct BeaverEdit {
  int kind;
  intptr_t index;
  MlirStringRef name;
  MlirAttribute attribute;
};
namespace {
class DeclarativePattern final : public mlir::RewritePattern {
  struct Check {
    int kind;
    intptr_t index, other;
    std::string name;
    mlir::Attribute attr;
    mlir::Type type;
  };
  struct Edit {
    int kind;
    intptr_t index;
    std::string name;
    mlir::Attribute attr;
  };
  std::vector<Check> checks;
  std::vector<Edit> edits;

public:
  DeclarativePattern(llvm::StringRef root, unsigned benefit, mlir::MLIRContext *context,
                     const BeaverConstraint *constraints, size_t n, const BeaverEdit *actions, size_t m)
      : RewritePattern(root, benefit, context) {
    for (size_t i = 0; i < n; ++i) {
      const auto &c = constraints[i];
      checks.push_back({c.kind, c.index, c.other, unwrap(c.name).str(), unwrap(c.attribute), unwrap(c.type)});
    }
    for (size_t i = 0; i < m; ++i) {
      const auto &a = actions[i];
      edits.push_back({a.kind, a.index, unwrap(a.name).str(), unwrap(a.attribute)});
    }
    setHasBoundedRewriteRecursion();
    setDebugName("beaver-declarative");
  }
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    auto operand = [&](intptr_t i) { return i >= 0 && static_cast<size_t>(i) < op->getNumOperands(); };
    for (const auto &c : checks) {
      bool matches = false;
      switch (c.kind) {
      case 0:
        matches = op->getNumOperands() == static_cast<size_t>(c.index);
        break;
      case 1:
        matches = op->getNumResults() == static_cast<size_t>(c.index);
        break;
      case 2:
        matches = op->getAttr(c.name) == c.attr;
        break;
      case 3:
        matches = operand(c.index) && op->getOperand(c.index).getType() == c.type;
        break;
      case 4:
        matches = c.index >= 0 && static_cast<size_t>(c.index) < op->getNumResults() &&
                  op->getResult(c.index).getType() == c.type;
        break;
      case 5:
        matches = operand(c.index) && operand(c.other) && op->getOperand(c.index) == op->getOperand(c.other);
        break;
      }
      if (!matches) return mlir::failure();
    }
    const auto &first = edits.front();
    if (first.kind == 1) {
      if (!operand(first.index) || op->getNumResults() != 1 ||
          op->getResult(0).getType() != op->getOperand(first.index).getType())
        return mlir::failure();
      rewriter.replaceOp(op, op->getOperand(first.index));
    } else if (first.kind == 2) {
      if (!op->use_empty()) return mlir::failure();
      rewriter.eraseOp(op);
    } else {
      bool changed = false;
      for (const auto &edit : edits) {
        auto previous = op->getAttr(edit.name);
        auto attribute = edit.attr;
        if (beaverOperationHasInherentAttributeByName(wrap(op), wrap(llvm::StringRef(edit.name))) &&
            (!previous || previous.getTypeID() != attribute.getTypeID()))
          return mlir::failure();
        changed |= previous != edit.attr;
      }
      if (!changed) return mlir::failure();
      rewriter.modifyOpInPlace(op, [&] {
        for (const auto &edit : edits) op->setAttr(edit.name, edit.attr);
      });
    }
    return mlir::success();
  }
};
class NoopPass final : public mlir::PassWrapper<NoopPass, mlir::OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NoopPass)
  void runOnOperation() override {} // Deliberately retains normal per-pass verification.
};
} // namespace

extern "C" MLIR_CAPI_EXPORTED void beaverDeclarativePatternAdd(MlirRewritePatternSet set, MlirContext context,
                                                               MlirStringRef root, unsigned benefit,
                                                               const BeaverConstraint *constraints, size_t n,
                                                               const BeaverEdit *actions, size_t m) {
  unwrap(set)->add<DeclarativePattern>(unwrap(root), benefit, unwrap(context), constraints, n, actions, m);
}
extern "C" MLIR_CAPI_EXPORTED MlirPass beaverCreateNoopPass() { return wrap(new NoopPass); }
