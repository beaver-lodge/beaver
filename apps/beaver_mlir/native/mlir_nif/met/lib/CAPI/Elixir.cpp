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

MLIR_CAPI_EXPORTED MlirLogicalResult beaverApplyOwnedPatternSet(
    MlirRegion region, MlirRewritePatternSet patternList) {
  return wrap(applyPatternsAndFoldGreedily(*unwrap(region),
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
