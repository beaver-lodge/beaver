#ifndef NATIVE_MLIR_C_INCLUDE_MLIR_CAPI_BEAVER_H_
#define NATIVE_MLIR_C_INCLUDE_MLIR_CAPI_BEAVER_H_

#include "mlir-c/Beaver/Context.h"
#include "mlir-c/Beaver/Op.h"
#include "mlir-c/Beaver/PDL.h"
#include "mlir-c/Beaver/Pass.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

DEFINE_C_API_PTR_METHODS(MlirRewritePatternSet, mlir::RewritePatternSet)
DEFINE_C_API_PTR_METHODS(MlirOperand, mlir::Value::use_iterator)
DEFINE_C_API_METHODS(MlirRegisteredOperationName, mlir::RegisteredOperationName)

#endif // NATIVE_MLIR_C_INCLUDE_MLIR_CAPI_BEAVER_H_
