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

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetDescription(MlirPass pass) {
  return wrap(unwrap(pass)->getDescription());
}

MLIR_CAPI_EXPORTED bool beaverIsOpNameTerminator(MlirStringRef op_name,
                                                 MlirContext context) {
  return OperationName(unwrap(op_name), unwrap(context))
      .mightHaveTrait<OpTrait::IsTerminator>();
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

MLIR_CAPI_EXPORTED void beaverEnterMultiThreadedExecution(MlirContext context) {
  unwrap(context)->enterMultiThreadedExecution();
}

MLIR_CAPI_EXPORTED void beaverExitMultiThreadedExecution(MlirContext context) {
  unwrap(context)->exitMultiThreadedExecution();
}

static_assert(sizeof(mlir::Value::use_iterator) < MlirOperandSize,
              "MlirOperandSize too small");
static inline MlirOperand wrap(mlir::Value::use_iterator cpp) {
  MlirOperand ret;
  memcpy(ret.buffer, &cpp, sizeof(mlir::Value::use_iterator));
  return ret;
}

static inline mlir::Value::use_iterator unwrap(MlirOperand c) {
  mlir::Value::use_iterator ret;
  memcpy(&ret, c.buffer, sizeof(mlir::Value::use_iterator));
  return ret;
}

MLIR_CAPI_EXPORTED MlirOperand beaverValueGetFirstOperand(MlirValue value) {
  return wrap(unwrap(value).use_begin());
}

MLIR_CAPI_EXPORTED MlirOperand beaverOperandGetNext(MlirOperand operand) {
  auto unwrapped = unwrap(operand);
  unwrapped++;
  return wrap(unwrapped);
}

MLIR_CAPI_EXPORTED bool beaverOperandIsNull(MlirOperand operand) {
  auto unwrapped = unwrap(operand);
  return unwrapped == nullptr || unwrapped == unwrap(operand)->get().use_end();
}

MLIR_CAPI_EXPORTED MlirValue beaverOperandGetValue(MlirOperand operand) {
  return wrap(unwrap(operand)->get());
}

MLIR_CAPI_EXPORTED MlirOperation beaverOperandGetOwner(MlirOperand operand) {
  return wrap((unwrap(operand))->getOwner());
}

MLIR_CAPI_EXPORTED uint32_t beaverOperandGetNumber(MlirOperand operand) {
  return unwrap(operand)->getOperandNumber();
}
