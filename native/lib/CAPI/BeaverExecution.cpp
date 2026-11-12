#include "mlir-c/Beaver/Context.h"
#include "mlir-c/Beaver/Pass.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/CAPI/Execution.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir-c/Beaver/Op.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Transforms/DialectConversion.h"
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

// A callback-free declarative rewrite and dialect conversion engine.
struct BeaverConstraint {
  int kind;
  intptr_t index, other;
  MlirStringRef name;
  MlirStringRef extra;
  MlirAttribute attribute;
  MlirType type;
};
struct BeaverEdit {
  int kind;
  intptr_t index, other;
  MlirStringRef name;
  MlirStringRef extra;
  MlirAttribute attribute;
  MlirType type;
};
namespace {
struct DeclarativeProgram {
  struct Check {
    int kind;
    intptr_t index, other;
    std::string name;
    mlir::Attribute attr;
    mlir::Type type;
  };
  struct Edit {
    int kind;
    intptr_t index, other;
    std::string name;
    std::string extra;
    mlir::Attribute attr;
    mlir::Type type;
  };
  std::vector<Check> checks;
  std::vector<Edit> edits;

  static std::shared_ptr<DeclarativeProgram> create(mlir::MLIRContext *context, const BeaverConstraint *constraints,
                                                    size_t n, const BeaverEdit *actions, size_t m) {
    auto prog = std::make_shared<DeclarativeProgram>();
    for (size_t i = 0; i < n; ++i) {
      const auto &c = constraints[i];
      mlir::Attribute attr = unwrap(c.attribute);
      mlir::Type type = unwrap(c.type);
      if (c.kind == 2 && !attr && c.extra.length > 0) {
        attr = mlir::parseAttribute(unwrap(c.extra), context);
        if (!attr) return nullptr;
      }
      if ((c.kind == 3 || c.kind == 4) && !type && c.extra.length > 0) {
        type = mlir::parseType(unwrap(c.extra), context);
        if (!type) return nullptr;
      }
      prog->checks.push_back({c.kind, c.index, c.other, unwrap(c.name).str(), attr, type});
    }
    for (size_t i = 0; i < m; ++i) {
      const auto &a = actions[i];
      mlir::Attribute attr = unwrap(a.attribute);
      mlir::Type type = unwrap(a.type);
      if ((a.kind == 0 || a.kind == 5) && !attr && a.extra.length > 0) {
        attr = mlir::parseAttribute(unwrap(a.extra), context);
        if (!attr) return nullptr;
      }
      if ((a.kind == 7 || a.kind == 10 || a.kind == 11 || a.kind == 12) && !type && a.name.length > 0) {
        type = mlir::parseType(unwrap(a.name), context);
        if (!type) return nullptr;
      }
      if (a.kind == 12) {
        if (!prog->edits.empty()) prog->edits.front().type = type;
        continue;
      }
      prog->edits.push_back({a.kind, a.index, a.other, unwrap(a.name).str(), unwrap(a.extra).str(), attr, type});
    }
    return prog;
  }

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ValueRange operands,
                                      const mlir::TypeConverter *converter, mlir::PatternRewriter &rewriter,
                                      mlir::ConversionPatternRewriter *convRewriter) const {
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
    auto convertOrKeep = [&](mlir::Type t) -> mlir::Type {
      if (!t || !converter) return t;
      mlir::Type converted = converter->convertType(t);
      return converted ? converted : t;
    };
    const auto &first = edits.front();
    if (first.kind == 1) {
      if (first.index < 0 || static_cast<size_t>(first.index) >= operands.size() || op->getNumResults() != 1)
        return mlir::failure();
      mlir::Value replacement = operands[first.index];
      if (!converter && op->getResult(0).getType() != replacement.getType()) return mlir::failure();
      rewriter.replaceOp(op, replacement);
    } else if (first.kind == 2) {
      if (!op->use_empty()) return mlir::failure();
      rewriter.eraseOp(op);
    } else if (first.kind == 3) {
      llvm::SmallVector<mlir::Value> newOperands;
      if (first.index == -1) {
        newOperands.append(operands.begin(), operands.end());
      } else {
        for (size_t i = 1; i < edits.size(); ++i) {
          if (edits[i].kind == 4) {
            if (edits[i].index < 0 || static_cast<size_t>(edits[i].index) >= operands.size()) return mlir::failure();
            newOperands.push_back(operands[edits[i].index]);
          }
        }
      }
      mlir::NamedAttrList newAttrs;
      if (first.other & 2) {
        for (const mlir::NamedAttribute &na : op->getAttrs()) newAttrs.set(na.getName(), na.getValue());
      } else if (first.other & 8) {
        for (const mlir::NamedAttribute &na : op->getAttrs()) {
          bool excluded = false;
          llvm::StringRef attrName = na.getName().getValue();
          for (size_t i = 1; i < edits.size(); ++i) {
            if (edits[i].kind == 13 && attrName == edits[i].name) {
              excluded = true;
              break;
            }
          }
          if (!excluded) newAttrs.set(na.getName(), na.getValue());
        }
      }
      for (size_t i = 1; i < edits.size(); ++i) {
        if (edits[i].kind == 6) {
          if (mlir::Attribute copied = op->getAttr(edits[i].name)) newAttrs.set(edits[i].name, copied);
        } else if (edits[i].kind == 5) {
          newAttrs.set(edits[i].name, edits[i].attr);
        } else if (edits[i].kind == 14) {
          mlir::Attribute src = op->getAttr(edits[i].extra);
          if (auto strAttr = llvm::dyn_cast_or_null<mlir::StringAttr>(src)) {
            newAttrs.set(edits[i].name, mlir::FlatSymbolRefAttr::get(op->getContext(), strAttr.getValue().ltrim('@')));
          } else if (auto symAttr = llvm::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(src)) {
            newAttrs.set(edits[i].name, symAttr);
          } else {
            return mlir::failure();
          }
        }
      }
      if (first.other & 4) {
        auto buildTargetFnType = [&](llvm::ArrayRef<mlir::Type> rawInputs,
                                     llvm::ArrayRef<mlir::Type> rawResults) -> mlir::Type {
          llvm::SmallVector<mlir::Type> convInputs;
          llvm::SmallVector<mlir::Type> convResults;
          if (converter) {
            mlir::TypeConverter::SignatureConversion sigConv(rawInputs.size());
            for (auto [idx, inputType] : llvm::enumerate(rawInputs)) {
              if (mlir::failed(converter->convertSignatureArg(idx, inputType, sigConv))) return mlir::Type();
            }
            llvm::append_range(convInputs, sigConv.getConvertedTypes());
            if (mlir::failed(converter->convertTypes(rawResults, convResults))) return mlir::Type();
          } else {
            llvm::append_range(convInputs, rawInputs);
            llvm::append_range(convResults, rawResults);
          }
          auto *fnCtx = op->getContext();
          if (first.name == "llvm.func") {
            mlir::Type llvmRet;
            if (convResults.empty()) {
              llvmRet = mlir::LLVM::LLVMVoidType::get(fnCtx);
            } else if (convResults.size() == 1) {
              llvmRet = convResults[0];
            } else {
              llvmRet = mlir::LLVM::LLVMStructType::getLiteral(fnCtx, convResults);
            }
            return mlir::LLVM::LLVMFunctionType::get(llvmRet, convInputs, false);
          }
          return mlir::FunctionType::get(fnCtx, convInputs, convResults);
        };

        mlir::Type newFnType;
        auto typeAttr = op->getAttrOfType<mlir::TypeAttr>("function_type");
        if (!(first.other & 16) && typeAttr) {
          if (auto fnType = llvm::dyn_cast<mlir::FunctionType>(typeAttr.getValue())) {
            newFnType = buildTargetFnType(fnType.getInputs(), fnType.getResults());
            if (!newFnType) return mlir::failure();
          }
        } else if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
          mlir::Region &body = op->getRegion(0);
          llvm::SmallVector<mlir::Type> inferredInputs(body.front().getArgumentTypes().begin(),
                                                       body.front().getArgumentTypes().end());
          llvm::SmallVector<mlir::Type> inferredResults;
          for (mlir::Block &blk : body) {
            if (blk.empty()) continue;
            mlir::Operation &term = blk.back();
            if ((term.hasTrait<mlir::OpTrait::IsTerminator>() || !term.isRegistered()) &&
                term.getNumSuccessors() == 0 && term.getNumResults() == 0) {
              llvm::append_range(inferredResults, term.getOperandTypes());
              break;
            }
          }
          newFnType = buildTargetFnType(inferredInputs, inferredResults);
          if (!newFnType) return mlir::failure();
        }
        if (newFnType) {
          newAttrs.set("function_type", mlir::TypeAttr::get(newFnType));
        }
      }
      llvm::SmallVector<mlir::Type> resultTypes;
      bool hasExplicitResults = false;
      for (size_t i = 1; i < edits.size(); ++i) {
        if (edits[i].kind == 7) {
          hasExplicitResults = true;
          if (edits[i].index == -1) break;
          mlir::Type t = convertOrKeep(edits[i].type);
          if (!t) return mlir::failure();
          resultTypes.push_back(t);
        }
      }
      if (!hasExplicitResults) {
        for (mlir::Type t : op->getResultTypes()) {
          mlir::Type converted = convertOrKeep(t);
          if (!converted) return mlir::failure();
          resultTypes.push_back(converted);
        }
      }
      mlir::OperationState state(op->getLoc(), first.name, newOperands, resultTypes, newAttrs.getAttrs());
      bool moveRegions = (first.other & 1) != 0;
      if (moveRegions) {
        for (unsigned r = 0; r < op->getNumRegions(); ++r) state.addRegion();
      }
      mlir::Operation *newOp = rewriter.create(state);
      if (moveRegions) {
        for (unsigned r = 0; r < op->getNumRegions(); ++r) {
          rewriter.inlineRegionBefore(op->getRegion(r), newOp->getRegion(r), newOp->getRegion(r).end());
          if ((first.other & 4) && converter && convRewriter && !newOp->getRegion(r).empty()) {
            if (mlir::failed(convRewriter->convertRegionTypes(&newOp->getRegion(r), *converter)))
              return mlir::failure();
          }
        }
      }
      if (newOp->getNumResults() == op->getNumResults()) {
        rewriter.replaceOp(op, newOp->getResults());
      } else if (op->use_empty()) {
        rewriter.eraseOp(op);
      } else {
        return mlir::failure();
      }
    } else if (first.kind == 8) {
      auto *ctx = op->getContext();
      auto loc = op->getLoc();
      auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
      auto i64Type = mlir::IntegerType::get(ctx, 64);
      bool usesStack = false;
      for (size_t i = 1; i < edits.size(); ++i) {
        if (edits[i].kind == 10 || edits[i].kind == 11) usesStack = true;
      }
      mlir::Value savedStack;
      mlir::Value one;
      if (usesStack) {
        if (first.index != 0) {
          mlir::OperationState saveState(loc, "llvm.intr.stacksave");
          saveState.addTypes(ptrType);
          savedStack = rewriter.create(saveState)->getResult(0);
        }
        one = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1)).getRes();
      }
      size_t opIdx = 0;
      llvm::SmallVector<mlir::Value> callOperands;
      llvm::SmallVector<std::pair<mlir::Value, mlir::Type>> outSlots;
      for (size_t i = 1; i < edits.size(); ++i) {
        const auto &param = edits[i];
        if (param.kind == 9) {
          if (opIdx >= operands.size()) return mlir::failure();
          callOperands.push_back(operands[opIdx++]);
        } else if (param.kind == 10) {
          if (opIdx >= operands.size()) return mlir::failure();
          mlir::Value val = operands[opIdx++];
          mlir::Type elemType = param.type ? convertOrKeep(param.type) : val.getType();
          if (!elemType) return mlir::failure();
          mlir::Value slot = mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, elemType, one).getRes();
          mlir::LLVM::StoreOp::create(rewriter, loc, val, slot);
          callOperands.push_back(slot);
        } else if (param.kind == 11) {
          mlir::Type elemType = param.type ? convertOrKeep(param.type) : mlir::Type();
          if (!elemType) return mlir::failure();
          mlir::Value slot = mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, elemType, one).getRes();
          if (param.index != 0) {
            mlir::Value zero = mlir::LLVM::ZeroOp::create(rewriter, loc, elemType).getRes();
            mlir::LLVM::StoreOp::create(rewriter, loc, zero, slot);
          }
          callOperands.push_back(slot);
          outSlots.push_back({slot, elemType});
        }
      }
      mlir::Type callRetType = first.type ? convertOrKeep(first.type) : mlir::Type();
      if (!first.extra.empty() && first.other != 0) {
        if (mlir::Operation *symTableOp = mlir::SymbolTable::getNearestSymbolTable(op)) {
          llvm::SmallVector<mlir::Type> paramTypes;
          for (mlir::Value v : callOperands) paramTypes.push_back(v.getType());
          auto expectedFnType = [&]() -> mlir::Type {
            if (first.name == "llvm.call") {
              mlir::Type llvmRet = callRetType ? callRetType : mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
              return mlir::LLVM::LLVMFunctionType::get(llvmRet, paramTypes, false);
            }
            if (first.name == "func.call") {
              llvm::SmallVector<mlir::Type> retTypes;
              if (callRetType) retTypes.push_back(callRetType);
              return mlir::FunctionType::get(ctx, paramTypes, retTypes);
            }
            return mlir::Type();
          };
          if (mlir::Operation *existingSym = mlir::SymbolTable::lookupSymbolIn(symTableOp, first.extra)) {
            if ((first.other & 2) != 0) {
              mlir::Type expType = expectedFnType();
              auto existingAttr = existingSym->getAttrOfType<mlir::TypeAttr>("function_type");
              if (expType && (!existingAttr || existingAttr.getValue() != expType)) {
                return mlir::failure();
              }
            }
          } else if ((first.other & 1) != 0) {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&symTableOp->getRegion(0).front());
            if (first.name == "llvm.call") {
              auto fnType = llvm::cast<mlir::LLVM::LLVMFunctionType>(expectedFnType());
              mlir::LLVM::LLVMFuncOp::create(rewriter, loc, first.extra, fnType, mlir::LLVM::Linkage::External);
            } else if (first.name == "func.call") {
              auto fnType = llvm::cast<mlir::FunctionType>(expectedFnType());
              auto funcOp = mlir::func::FuncOp::create(rewriter, loc, first.extra, fnType);
              funcOp.setPrivate();
            }
          }
        }
      }
      mlir::Operation *callOp = nullptr;
      if (first.name == "llvm.call" && !first.extra.empty()) {
        llvm::SmallVector<mlir::Type> callResTypes;
        if (callRetType) callResTypes.push_back(callRetType);
        callOp = mlir::LLVM::CallOp::create(rewriter, loc, callResTypes, mlir::FlatSymbolRefAttr::get(ctx, first.extra),
                                            callOperands);
      } else {
        mlir::OperationState callState(loc, first.name);
        callState.addOperands(callOperands);
        if (callRetType) callState.addTypes(callRetType);
        if (!first.extra.empty()) callState.addAttribute("callee", mlir::FlatSymbolRefAttr::get(ctx, first.extra));
        callOp = rewriter.create(callState);
      }
      llvm::SmallVector<mlir::Value> results;
      if (callOp->getNumResults() > 0 && op->getNumResults() == callOp->getNumResults() + outSlots.size()) {
        results.append(callOp->result_begin(), callOp->result_end());
      }
      for (const auto &pair : outSlots) {
        results.push_back(mlir::LLVM::LoadOp::create(rewriter, loc, pair.second, pair.first).getRes());
      }
      if (results.empty() && callOp->getNumResults() == op->getNumResults()) {
        results.append(callOp->result_begin(), callOp->result_end());
      }
      if (savedStack) {
        mlir::OperationState restoreState(loc, "llvm.intr.stackrestore");
        restoreState.addOperands(savedStack);
        rewriter.create(restoreState);
      }
      if (op->getNumResults() == 0) {
        rewriter.eraseOp(op);
      } else if (results.size() == op->getNumResults()) {
        rewriter.replaceOp(op, results);
      } else {
        return mlir::failure();
      }
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

class DeclarativePattern final : public mlir::RewritePattern {
  std::shared_ptr<DeclarativeProgram> program;

public:
  DeclarativePattern(llvm::StringRef root, unsigned benefit, mlir::MLIRContext *context,
                     std::shared_ptr<DeclarativeProgram> program)
      : RewritePattern(root, benefit, context), program(std::move(program)) {
    setHasBoundedRewriteRecursion();
    setDebugName("beaver-declarative");
  }
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    return program->matchAndRewrite(op, op->getOperands(), nullptr, rewriter, nullptr);
  }
};

class DeclarativeConversionPattern final : public mlir::ConversionPattern {
  std::shared_ptr<DeclarativeProgram> program;

public:
  DeclarativeConversionPattern(const mlir::TypeConverter &converter, llvm::StringRef root, unsigned benefit,
                               mlir::MLIRContext *context, std::shared_ptr<DeclarativeProgram> program)
      : ConversionPattern(converter, root, benefit, context), program(std::move(program)) {
    setHasBoundedRewriteRecursion();
    setDebugName("beaver-declarative-conversion");
  }
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    return program->matchAndRewrite(op, operands, getTypeConverter(), rewriter, &rewriter);
  }
};

class NoopPass final : public mlir::PassWrapper<NoopPass, mlir::OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NoopPass)
  void runOnOperation() override {} // Deliberately retains normal per-pass verification.
};
} // namespace

extern "C" MLIR_CAPI_EXPORTED bool beaverDeclarativePatternAdd(MlirRewritePatternSet set, MlirContext context,
                                                               MlirTypeConverter typeConverter, MlirStringRef root,
                                                               unsigned benefit, const BeaverConstraint *constraints,
                                                               size_t n, const BeaverEdit *actions, size_t m) {
  auto prog = DeclarativeProgram::create(unwrap(context), constraints, n, actions, m);
  if (!prog) return false;
  if (typeConverter.ptr) {
    unwrap(set)->add<DeclarativeConversionPattern>(*unwrap(typeConverter), unwrap(root), benefit, unwrap(context),
                                                   std::move(prog));
  } else {
    unwrap(set)->add<DeclarativePattern>(unwrap(root), benefit, unwrap(context), std::move(prog));
  }
  return true;
}

extern "C" MLIR_CAPI_EXPORTED bool beaverRewritePatternSetAddPDL(MlirRewritePatternSet set, MlirOperation pdlModuleOp) {
  if (!set.ptr || !pdlModuleOp.ptr) return false;
  auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(unwrap(pdlModuleOp));
  if (!moduleOp) return false;
  if (mlir::failed(mlir::verify(moduleOp.getOperation()))) return false;
  mlir::OwningOpRef<mlir::ModuleOp> cloned(llvm::cast<mlir::ModuleOp>(moduleOp.getOperation()->clone()));
  mlir::PDLPatternModule pdlPattern(std::move(cloned));
  unwrap(set)->add(std::move(pdlPattern));
  return true;
}

extern "C" MLIR_CAPI_EXPORTED MlirPass beaverCreateNoopPass() { return wrap(new NoopPass); }
