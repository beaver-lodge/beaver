#include "mlir-c/Beaver/ActionTracing.h"
#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Debug/BreakpointManagers/TagBreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Unit.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

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

/// Serialize one IR unit into a compact JSON object without materializing full
/// IR text.
static llvm::json::Object serializeIRUnit(const IRUnit &unit) {
  if (isa<Operation *>(unit)) {
    auto *op = cast<Operation *>(unit);
    return llvm::json::Object{{"kind", "operation"},
                              {"name", op->getName().getStringRef().str()},
                              {"loc", printLocation(op->getLoc())}};
  }
  if (isa<Region *>(unit))
    return llvm::json::Object{{"kind", "region"}};
  if (isa<Block *>(unit))
    return llvm::json::Object{{"kind", "block"}};
  if (isa<Value>(unit)) {
    Value value = cast<Value>(unit);
    return llvm::json::Object{{"kind", "value"},
                              {"loc", printLocation(value.getLoc())}};
  }
  return llvm::json::Object{{"kind", "unknown"}};
}

/// Serialize the IR units associated with an action.
static llvm::json::Array serializeIRUnits(ArrayRef<IRUnit> units) {
  llvm::json::Array result;
  result.reserve(units.size());
  for (const IRUnit &unit : units)
    result.emplace_back(serializeIRUnit(unit));
  return result;
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
    std::vector<llvm::json::Object> drained;
    {
      std::lock_guard<std::mutex> lock(mutex);
      drained.swap(events);
    }
    llvm::json::Array payload;
    payload.reserve(drained.size());
    for (auto &event : drained)
      payload.emplace_back(std::move(event));

    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    os << llvm::json::Value(std::move(payload));
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
      session->push(llvm::json::Object{
          {"phase", "before"},
          {"tag", action.getTag().str()},
          {"depth", static_cast<uint64_t>(stack->getDepth())},
          {"description", std::move(description)},
          {"ir_units", serializeIRUnits(action.getContextIRUnits())},
          {"t_ns", nowNs()}});
    }

    void afterExecute(const ActionActiveStack *stack) override {
      const Action &action = stack->getAction();
      if (!session->shouldObserve(action))
        return;
      session->push(llvm::json::Object{
          {"phase", "after"},
          {"tag", action.getTag().str()},
          {"depth", static_cast<uint64_t>(stack->getDepth())},
          {"t_ns", nowNs()}});
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

  void push(llvm::json::Object event) {
    std::lock_guard<std::mutex> lock(mutex);
    events.push_back(std::move(event));
  }

  static std::vector<std::string> parseTagList(llvm::StringRef json) {
    return parseStringList(json);
  }

  static std::vector<std::string> parseStringList(llvm::StringRef json) {
    if (json.empty())
      return {};
    auto value = llvm::json::parse(json);
    if (!value) {
      llvm::consumeError(value.takeError());
      return {};
    }
    const auto *array = value->getAsArray();
    if (!array)
      return {};

    std::vector<std::string> result;
    result.reserve(array->size());
    for (const auto &element : *array) {
      auto string = element.getAsString();
      if (!string)
        return {};
      result.push_back(string->str());
    }
    return result;
  }

  static std::unordered_map<std::string, uint64_t>
  parseCountMap(llvm::StringRef json) {
    std::unordered_map<std::string, uint64_t> result;
    if (json.empty())
      return result;
    auto value = llvm::json::parse(json);
    if (!value) {
      llvm::consumeError(value.takeError());
      return result;
    }
    const auto *object = value->getAsObject();
    if (!object)
      return result;

    for (const auto &entry : *object) {
      auto count = entry.second.getAsUINT64();
      if (!count)
        return {};
      result.emplace(entry.first.str(), *count);
    }
    return result;
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
  std::vector<llvm::json::Object> events;
};

} // namespace

MLIR_CAPI_EXPORTED MlirBeaverActionTracing mlirBeaverActionTracingAttach(
    MlirContext context, MlirStringRef filter_json, MlirStringRef location_json,
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

MLIR_CAPI_EXPORTED bool
mlirBeaverActionTracingDrain(MlirBeaverActionTracing tracing,
                             MlirBeaverActionEventsCallback callback,
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
