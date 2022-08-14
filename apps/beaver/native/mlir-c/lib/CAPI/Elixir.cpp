#include "mlir-c/Dialect/Elixir.h"

#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Elixir/IR/ElixirDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Elixir, elixir,
                                      mlir::elixir::ElixirDialect)

// TODO: move these to another file
#include "mlir/CAPI/Beaver.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

MLIR_CAPI_EXPORTED MlirPDLPatternModule beaverPDLPatternGet(MlirModule module) {
  // should this module be removed from parent?
  auto *pdlPattern = new PDLPatternModule(unwrap(module));
  return wrap(pdlPattern);
}

MLIR_CAPI_EXPORTED MlirRewritePatternSet
beaverRewritePatternSetGet(MlirContext context) {
  return wrap(new mlir::RewritePatternSet(unwrap(context)));
}

MLIR_CAPI_EXPORTED MlirRewritePatternSet beaverPatternSetAddOwnedPDLPattern(
    MlirRewritePatternSet patternList, MlirPDLPatternModule pdlPattern) {
  auto &set = unwrap(patternList)->add(std::move(*(unwrap(pdlPattern))));
  return wrap(&set);
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverApplyOwnedPatternSetOnRegion(
    MlirRegion region, MlirRewritePatternSet patternList) {
  return wrap(applyPatternsAndFoldGreedily(*unwrap(region),
                                           std::move(*unwrap(patternList))));
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverApplyOwnedPatternSetOnOperation(
    MlirOperation op, MlirRewritePatternSet patternList) {
  return wrap(applyPatternsAndFoldGreedily(unwrap(op),
                                           std::move(*unwrap(patternList))));
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

MLIR_CAPI_EXPORTED bool beaverIsOpNameTerminator(MlirStringRef op_name,
                                                 MlirContext context) {
  auto name = OperationName(unwrap(op_name), unwrap(context));
  return name.isRegistered() && name.mightHaveTrait<OpTrait::IsTerminator>();
}

MLIR_CAPI_EXPORTED intptr_t
beaverGetNumRegisteredOperations(MlirContext context) {
  return unwrap(context)->getRegisteredOperations().size();
}

MLIR_CAPI_EXPORTED MlirRegisteredOperationName
beaverGetRegisteredOperationName(MlirContext context, intptr_t pos) {
  mlir::RegisteredOperationName name =
      unwrap(context)->getRegisteredOperations()[pos];
  return wrap(name);
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverRegisteredOperationNameGetDialectName(MlirRegisteredOperationName name) {
  return wrap(unwrap(name).getDialectNamespace());
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverRegisteredOperationNameGetOpName(MlirRegisteredOperationName name) {
  return wrap(unwrap(name).stripDialect());
}

MLIR_CAPI_EXPORTED void
beaverRegisteredOperationsOfDialect(MlirContext context, MlirStringRef dialect,
                                    MlirRegisteredOperationName *ret,
                                    size_t *num) {
  int i = 0;
  for (auto &op : unwrap(context)->getRegisteredOperations()) {
    if (std::string(op.getDialectNamespace()) == std::string(unwrap(dialect))) {
      if (i > 300) {
        llvm::errs() << "dialect " << unwrap(dialect) << " has more than 300 "
                     << "operations\n";
        exit(1);
      }
      ret[i] = wrap(op);
      i += 1;
    }
  }
  *num = i;
}

MLIR_CAPI_EXPORTED void
beaverRegisteredDialects(MlirContext context, MlirStringRef *ret, size_t *num) {
  int i = 0;
  for (auto dialect : unwrap(context)->getDialectRegistry().getDialectNames()) {
    if (i > 300) {
      llvm::errs() << "more than 300 dialect in registry" << dialect;
      exit(1);
    }
    ret[i] = wrap(dialect);
    i += 1;
  }
  *num = i;
}

MLIR_CAPI_EXPORTED void beaverEnterMultiThreadedExecution(MlirContext context) {
  unwrap(context)->enterMultiThreadedExecution();
}

MLIR_CAPI_EXPORTED void beaverExitMultiThreadedExecution(MlirContext context) {
  unwrap(context)->exitMultiThreadedExecution();
}

MLIR_CAPI_EXPORTED MlirOperand beaverValueGetFirstOperand(MlirValue value) {
  // TODO: fix the leakage here
  auto *iter = new mlir::Value::use_iterator();
  *iter = unwrap(value).use_begin();
  return wrap(iter);
}

MLIR_CAPI_EXPORTED MlirOperand beaverOperandGetNext(MlirOperand operand) {
  auto iter = new mlir::Value::use_iterator();
  auto value = unwrap(operand);
  *iter = *value;
  (*iter)++;
  return wrap(iter);
}

MLIR_CAPI_EXPORTED bool beaverOperandIsNull(MlirOperand operand) {
  auto unwrapped = unwrap(operand);
  return operand.ptr == nullptr || (*unwrapped) == nullptr ||
         (*unwrapped) == (*unwrap(operand))->get().use_end();
}

MLIR_CAPI_EXPORTED MlirValue beaverOperandGetValue(MlirOperand operand) {
  return wrap((*unwrap(operand))->get());
}

MLIR_CAPI_EXPORTED MlirOperation beaverOperandGetOwner(MlirOperand operand) {
  return wrap((*unwrap(operand))->getOwner());
}

MLIR_CAPI_EXPORTED intptr_t beaverOperandGetNumber(MlirOperand operand) {
  return (*unwrap(operand))->getOperandNumber();
}

MLIR_CAPI_EXPORTED const char *
beaverStringRefGetData(MlirStringRef string_ref) {
  return string_ref.data;
}

MLIR_CAPI_EXPORTED size_t beaverStringRefGetLength(MlirStringRef string_ref) {
  return string_ref.length;
}

MLIR_CAPI_EXPORTED bool beaverContextIsNull(MlirContext w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverDialectIsNull(MlirDialect w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverDialectRegistryIsNull(MlirDialectRegistry w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverLocationIsNull(MlirLocation w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverModuleIsNull(MlirModule w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverOperationIsNull(MlirOperation w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverRegionIsNull(MlirRegion w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverBlockIsNull(MlirBlock w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverValueIsNull(MlirValue w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverTypeIsNull(MlirType w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverAttributeIsNull(MlirAttribute w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverSymbolTableIsNull(MlirSymbolTable w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverMlirExecutionEngineIsNull(MlirExecutionEngine w) {
  return !w.ptr;
}

MLIR_CAPI_EXPORTED MlirLocation
beaverMlirOperationStateGetLocation(MlirOperationState state) {
  return state.location;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumResults(MlirOperationState state) {
  return state.nResults;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumOperands(MlirOperationState state) {
  return state.nOperands;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumRegions(MlirOperationState state) {
  return state.nRegions;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumAttributes(MlirOperationState state) {
  return state.nAttributes;
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverMlirOperationStateGetName(MlirOperationState state) {
  return state.name;
}

MLIR_CAPI_EXPORTED MlirContext
beaverMlirOperationStateGetContext(MlirOperationState state) {
  return mlirLocationGetContext(state.location);
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsSuccess(MlirLogicalResult res) {
  return mlirLogicalResultIsSuccess(res);
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsFailure(MlirLogicalResult res) {
  return mlirLogicalResultIsFailure(res);
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultSuccess() {
  return mlirLogicalResultSuccess();
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultFailure() {
  return mlirLogicalResultFailure();
}

MLIR_CAPI_EXPORTED MlirIdentifier beaverOperationGetName(MlirOperation op,
                                                         intptr_t pos) {
  return mlirOperationGetAttribute(op, pos).name;
}

MLIR_CAPI_EXPORTED MlirAttribute beaverOperationGetAttribute(MlirOperation op,
                                                             intptr_t pos) {
  return mlirOperationGetAttribute(op, pos).attribute;
}

MLIR_CAPI_EXPORTED
MlirIdentifier beaverMlirNamedAttributeGetName(MlirNamedAttribute na) {
  return na.name;
}

MLIR_CAPI_EXPORTED
MlirAttribute beaverMlirNamedAttributeGetAttribute(MlirNamedAttribute na) {
  return na.attribute;
}

MLIR_CAPI_EXPORTED
void beaverOperationStateAddAttributes(MlirContext context,
                                       MlirOperationState *state, intptr_t n,
                                       MlirStringRef const *names,
                                       MlirAttribute const *attributes) {
  if (state->nAttributes != 0 || state->attributes != nullptr) {
    llvm::errs() << "attributes already set/n";
    exit(1);
  }

  MlirNamedAttribute na_arr[n];
  for (intptr_t i = 0; i < n; ++i) {
    auto attr = unwrap(attributes[i]);
    auto name = mlirIdentifierGet(context, names[i]);
    auto na = MlirNamedAttribute{name, wrap(attr)};
    na_arr[i] = na;
  }
  mlirOperationStateAddAttributes(state, n, na_arr);
}

MLIR_CAPI_EXPORTED MlirPass beaverCreateExternalPass(
    void (*construct)(void *userData), MlirTypeID passID, MlirStringRef name,
    MlirStringRef argument, MlirStringRef description, MlirStringRef opName,
    intptr_t nDependentDialects, MlirDialectHandle *dependentDialects,
    void (*destruct)(void *userData),
    MlirLogicalResult (*initialize)(MlirContext ctx, void *userData),
    void *(*clone)(void *userData),
    void (*run)(MlirOperation op, MlirExternalPass pass, void *userData),
    void *userData) {
  return mlirCreateExternalPass(
      passID, name, argument, description, opName, nDependentDialects,
      dependentDialects,
      MlirExternalPassCallbacks{construct, destruct, initialize, clone, run},
      userData);
}
