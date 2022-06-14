#ifndef APPS_MLIR_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_CAPI_BEAVER_H_
#define APPS_MLIR_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_CAPI_BEAVER_H_

#include "mlir-c/Beaver/Op.h"
#include "mlir-c/Beaver/PDL.h"
#include "mlir-c/Beaver/Pass.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

DEFINE_C_API_PTR_METHODS(MlirPDLPatternModule, mlir::PDLPatternModule)
DEFINE_C_API_PTR_METHODS(MlirRewritePatternSet, mlir::RewritePatternSet)

#endif // APPS_MLIR_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_CAPI_BEAVER_H_
