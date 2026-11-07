#include "mlir-c/Beaver/Remarks.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Regex.h"
#include "mlir-c/Remarks.h"

using namespace mlir;

namespace {
struct Receiver {
  MlirStringCallback callback;
  void *data;
  void (*destroy)(void *);
  ~Receiver() { destroy(data); }
};
static void report(MlirRemark remark, void *data) {
  auto *receiver = static_cast<Receiver *>(data);
  llvm::json::Array args;
  for (intptr_t i = 0; i < mlirRemarkGetNumArgs(remark); ++i)
    args.push_back(llvm::json::Array{unwrap(mlirRemarkGetArgKey(remark, i)).str(),
                                    unwrap(mlirRemarkGetArgValue(remark, i)).str()});
  std::string location;
  llvm::raw_string_ostream locationStream(location);
  unwrap(mlirRemarkGetLocation(remark)).print(locationStream);
  const char *kinds[] = {"unknown", "passed", "missed", "failure", "analysis"};
  llvm::json::Object payload{{"kind", kinds[mlirRemarkGetKind(remark)]},
      {"name", unwrap(mlirRemarkGetRemarkName(remark)).str()},
      {"category", unwrap(mlirRemarkGetFullCategoryName(remark)).str()},
      {"function", unwrap(mlirRemarkGetFunctionName(remark)).str()},
      {"location", location}, {"id", mlirRemarkGetId(remark)}, {"args", std::move(args)}};
  std::string json;
  llvm::raw_string_ostream(json) << llvm::json::Value(std::move(payload));
  receiver->callback(wrap(llvm::StringRef(json)), receiver->data);
}
static llvm::StringRef field(const llvm::json::Object &object, llvm::StringRef name) {
  return object.getString(name).value_or("");
}
} // namespace

MLIR_CAPI_EXPORTED bool beaverContextEnableOptimizationRemarks(
    MlirContext context, MlirStringRef filters, bool finalPolicy,
    MlirStringCallback callback, void *userData, void (*deleteUserData)(void *)) {
  std::unique_ptr<Receiver> receiver(new Receiver{callback, userData, deleteUserData});
  auto parsed = llvm::json::parse(unwrap(filters));
  if (!parsed) { llvm::consumeError(parsed.takeError()); return false; }
  auto *object = parsed->getAsObject();
  if (!object) return false;
  for (const auto &entry : *object) {
    auto pattern = entry.second.getAsString();
    if (!pattern || !llvm::Regex(*pattern).isValid()) return false;
  }
  MlirRemarkCategories categories{wrap(field(*object, "all")), wrap(field(*object, "passed")),
      wrap(field(*object, "missed")), wrap(field(*object, "analysis")), wrap(field(*object, "failed"))};
  return mlirLogicalResultIsSuccess(mlirContextEnableOptimizationRemarksWithCallback(
      context, categories, finalPolicy ? MlirRemarkPolicyFinal : MlirRemarkPolicyAll,
      report, receiver.release(), [](void *data) { delete static_cast<Receiver *>(data); }, false));
}

MLIR_CAPI_EXPORTED void beaverContextFinalizeOptimizationRemarks(MlirContext context) {
  mlirContextFinalizeOptimizationRemarks(context);
}

MLIR_CAPI_EXPORTED bool beaverContextHasOptimizationRemarks(MlirContext context) {
  return mlirContextHasRemarkEngine(context);
}

MLIR_CAPI_EXPORTED bool beaverEmitOptimizationRemark(MlirLocation location, MlirStringRef payload) {
  auto parsed = llvm::json::parse(unwrap(payload));
  if (!parsed) { llvm::consumeError(parsed.takeError()); return false; }
  auto *object = parsed->getAsObject();
  if (!object) return false;
  int64_t kind = object->getInteger("kind").value_or(0);
  if (kind < 1 || kind > 4) return false;
  SmallVector<MlirStringRef> keys, values;
  if (auto *args = object->getArray("args")) {
    for (const auto &arg : *args) {
      auto *pair = arg.getAsArray();
      if (!pair || pair->size() != 2 || !(*pair)[0].getAsString() || !(*pair)[1].getAsString()) return false;
      keys.push_back(wrap(*(*pair)[0].getAsString()));
      values.push_back(wrap(*(*pair)[1].getAsString()));
    }
  }
  return mlirEmitOptimizationRemark(location, static_cast<MlirRemarkKind>(kind),
      wrap(field(*object, "name")), wrap(field(*object, "category")), wrap(field(*object, "subcategory")),
      wrap(field(*object, "function")), wrap(field(*object, "message")), keys.size(), keys.data(), values.data());
}
