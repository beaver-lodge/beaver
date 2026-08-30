#ifndef MLIR_C_BEAVER_CONVERSION_H
#define MLIR_C_BEAVER_CONVERSION_H

#include "mlir-c/Rewrite.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Adds Beaver's frozen C++ Stage 0 bootstrap seed for Batata's restricted
/// compiler-kernel source. This is not a production Ex provider and must not
/// grow with runtime or standard-library semantics. The pattern set retains a
/// borrowed reference to `typeConverter`; callers must destroy the patterns
/// before destroying the converter, matching MLIR's normal conversion
/// lifetime contract.
MLIR_CAPI_EXPORTED void beaverPopulateExScalarConversionPatterns(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_CONVERSION_H
