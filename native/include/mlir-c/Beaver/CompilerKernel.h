#ifndef MLIR_C_BEAVER_COMPILER_KERNEL_H
#define MLIR_C_BEAVER_COMPILER_KERNEL_H

#include "mlir-c/Rewrite.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MLIR_BEAVER_COMPILER_KERNEL_ABI_VERSION 1u

typedef struct MlirBeaverCompilerKernelHostAPI
    MlirBeaverCompilerKernelHostAPI;

/// A callback-free native rewrite entrypoint owned by an external compiler
/// kernel. The callback runs synchronously on MLIR's conversion worker and
/// must not enter the BEAM. `diagnostic` may be used to attach a bounded error
/// message when returning failure.
typedef MlirLogicalResult (*MlirBeaverCompilerKernelRewriteFn)(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, void *userData,
    MlirStringCallback diagnostic, void *diagnosticUserData);

typedef void (*MlirBeaverCompilerKernelDestroyFn)(void *userData);

/// One versioned pattern descriptor. `structSize` must be set to
/// `sizeof(MlirBeaverCompilerKernelPattern)`. Ownership of `userData` passes
/// to Beaver only when `addPattern` succeeds.
typedef struct {
  size_t structSize;
  MlirStringRef name;
  MlirStringRef root;
  MlirStringRef version;
  unsigned benefit;
  MlirBeaverCompilerKernelRewriteFn matchAndRewrite;
  MlirBeaverCompilerKernelDestroyFn destroy;
  void *userData;
} MlirBeaverCompilerKernelPattern;

/// Append-only host function table passed to compiler-kernel artifacts.
/// Consumers must gate access by both `abiVersion` and `structSize`.
struct MlirBeaverCompilerKernelHostAPI {
  uint32_t abiVersion;
  size_t structSize;
  MlirLogicalResult (*addPattern)(
      void *hostContext, MlirRewritePatternSet patterns,
      MlirTypeConverter typeConverter,
      const MlirBeaverCompilerKernelPattern *pattern);
  void (*eraseOperation)(MlirConversionPatternRewriter rewriter,
                         MlirOperation operation);
};

typedef uint32_t (*MlirBeaverCompilerKernelABIVersionFn)(void);
typedef MlirStringRef (*MlirBeaverCompilerKernelManifestFn)(void);
typedef MlirLogicalResult (*MlirBeaverCompilerKernelPopulateFn)(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter,
    const MlirBeaverCompilerKernelHostAPI *host, void *hostContext,
    MlirStringCallback diagnostic, void *diagnosticUserData);

/// Permanently loads a content-addressed compiler-kernel artifact, verifies
/// its ABI and embedded non-self-referential manifest identity, calls its
/// population entrypoint, and verifies the exact registered pattern list.
///
/// Returns an empty string on success. Failure returns a stable protocol code,
/// a `|`, and a bounded human-readable message. The returned storage remains
/// valid until the next call on the same native thread and must not be freed.
MLIR_CAPI_EXPORTED MlirStringRef beaverCompilerKernelLoadAndPopulate(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter,
    MlirStringRef artifactPath, MlirStringRef abiVersionSymbol,
    MlirStringRef manifestSymbol, MlirStringRef populateSymbol,
    MlirStringRef expectedIdentity, MlirStringRef expectedPatternsJSON);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_COMPILER_KERNEL_H
