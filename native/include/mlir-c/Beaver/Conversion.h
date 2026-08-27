#ifndef MLIR_C_BEAVER_CONVERSION_H
#define MLIR_C_BEAVER_CONVERSION_H

#include "mlir-c/Rewrite.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Adds callback-free conversion patterns for high-frequency scalar and
/// control-flow operations in Beaver's dynamic `ex` dialect. The pattern set
/// retains a borrowed reference to `typeConverter`; callers must destroy the
/// patterns before destroying the converter, matching MLIR's normal
/// conversion lifetime contract.
MLIR_CAPI_EXPORTED void beaverPopulateExScalarConversionPatterns(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter);

/// Adds callback-free patterns that construct and query values through the
/// `ex.term` runtime ABI. Runtime declarations are inserted at module scope
/// and reused by symbol name.
MLIR_CAPI_EXPORTED void beaverPopulateExRuntimeConversionPatterns(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_CONVERSION_H
