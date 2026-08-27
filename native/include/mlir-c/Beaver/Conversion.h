#ifndef MLIR_C_BEAVER_CONVERSION_H
#define MLIR_C_BEAVER_CONVERSION_H

#include "mlir-c/Rewrite.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Adds callback-free conversion patterns for the high-frequency scalar
/// operations in Beaver's dynamic `ex` dialect. The pattern set retains a
/// borrowed reference to `typeConverter`; callers must destroy the patterns
/// before destroying the converter, matching MLIR's normal conversion
/// lifetime contract.
MLIR_CAPI_EXPORTED void beaverPopulateExScalarConversionPatterns(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_CONVERSION_H
