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
/// message when returning failure. All MLIR handles are borrowed for this
/// invocation and must not be retained or moved to another thread.
typedef MlirLogicalResult (*MlirBeaverCompilerKernelRewriteFn)(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, MlirTypeConverter typeConverter,
    void *userData,
    MlirStringCallback diagnostic, void *diagnosticUserData);

typedef void (*MlirBeaverCompilerKernelDestroyFn)(void *userData);

/// Scalar operation construction request. Every referenced handle is borrowed
/// for the duration of `createOperation`; the created operation is owned by
/// the conversion rewriter. Regions and successors are intentionally absent
/// from the descriptor. Regions can only be created and transferred through
/// the checked region host calls below; successors remain unsupported.
typedef struct {
  size_t structSize;
  MlirStringRef name;
  MlirLocation location;
  intptr_t nOperands;
  const MlirValue *operands;
  intptr_t nResultTypes;
  const MlirType *resultTypes;
  intptr_t nAttributes;
  const MlirNamedAttribute *attributes;
} MlirBeaverCompilerKernelOperation;

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
/// Consumers must gate access by both `abiVersion` and `structSize`. Every
/// function executes synchronously on the calling conversion worker; handles
/// remain borrowed and context-bound, and no function enters the BEAM.
struct MlirBeaverCompilerKernelHostAPI {
  uint32_t abiVersion;
  size_t structSize;
  MlirLogicalResult (*addPattern)(
      void *hostContext, MlirRewritePatternSet patterns,
      MlirTypeConverter typeConverter,
      const MlirBeaverCompilerKernelPattern *pattern);
  MlirLogicalResult (*operationResult)(MlirOperation operation, intptr_t index,
                                       MlirValue *result,
                                       MlirStringCallback diagnostic,
                                       void *diagnosticUserData);
  MlirLogicalResult (*operationLocation)(MlirOperation operation,
                                         MlirLocation *location,
                                         MlirStringCallback diagnostic,
                                         void *diagnosticUserData);
  MlirLogicalResult (*valueType)(MlirValue value, MlirType *type,
                                 MlirStringCallback diagnostic,
                                 void *diagnosticUserData);
  MlirLogicalResult (*convertType)(MlirTypeConverter converter, MlirType type,
                                   MlirType *converted,
                                   MlirStringCallback diagnostic,
                                   void *diagnosticUserData);
  MlirLogicalResult (*createOperation)(
      MlirConversionPatternRewriter rewriter,
      const MlirBeaverCompilerKernelOperation *operation,
      MlirOperation *created, MlirStringCallback diagnostic,
      void *diagnosticUserData);
  MlirLogicalResult (*replaceOperationWithValues)(
      MlirConversionPatternRewriter rewriter, MlirOperation operation,
      intptr_t nValues, const MlirValue *values,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*eraseOperation)(MlirConversionPatternRewriter rewriter,
                                      MlirOperation operation,
                                      MlirStringCallback diagnostic,
                                      void *diagnosticUserData);
  MlirLogicalResult (*operationAttribute)(
      MlirOperation operation, MlirStringRef name, MlirAttribute *attribute,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*attributeStringValue)(
      MlirAttribute attribute, MlirStringRef *value,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*integerType)(MlirConversionPatternRewriter rewriter,
                                   unsigned width, MlirType *type,
                                   MlirStringCallback diagnostic,
                                   void *diagnosticUserData);
  MlirLogicalResult (*integerAttribute)(
      MlirType type, int64_t value, MlirAttribute *attribute,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*namedAttribute)(
      MlirConversionPatternRewriter rewriter, MlirStringRef name,
      MlirAttribute attribute, MlirNamedAttribute *namedAttribute,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*operationCounts)(
      MlirOperation operation, intptr_t *nOperands, intptr_t *nResults,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*flatSymbolRefAttribute)(
      MlirConversionPatternRewriter rewriter, MlirStringRef symbol,
      MlirAttribute *attribute, MlirStringCallback diagnostic,
      void *diagnosticUserData);
  MlirLogicalResult (*ensureFunctionDeclaration)(
      MlirOperation anchor, MlirConversionPatternRewriter rewriter,
      MlirStringRef symbol, intptr_t nInputTypes, const MlirType *inputTypes,
      intptr_t nResultTypes, const MlirType *resultTypes,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  /// Append-only region/function inspection and construction surface. Region
  /// ownership transfer is coupled to replacement so an external callback
  /// cannot return failure after leaving the source operation bodyless.
  MlirLogicalResult (*operationOperand)(
      MlirOperation operation, intptr_t index, MlirValue *operand,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*attributeIntegerValue)(
      MlirAttribute attribute, int64_t *value, MlirStringCallback diagnostic,
      void *diagnosticUserData);
  MlirLogicalResult (*singleRegionBlock)(
      MlirOperation operation, intptr_t regionIndex, MlirBlock *block,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*blockArgumentCount)(
      MlirBlock block, intptr_t *count, MlirStringCallback diagnostic,
      void *diagnosticUserData);
  MlirLogicalResult (*blockArgument)(
      MlirBlock block, intptr_t index, MlirValue *argument,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*blockTerminator)(
      MlirBlock block, MlirOperation *terminator,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*functionType)(
      MlirConversionPatternRewriter rewriter, intptr_t nInputTypes,
      const MlirType *inputTypes, intptr_t nResultTypes,
      const MlirType *resultTypes, MlirType *type,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*typeAttribute)(
      MlirType type, MlirAttribute *attribute, MlirStringCallback diagnostic,
      void *diagnosticUserData);
  MlirLogicalResult (*createOperationWithRegions)(
      MlirConversionPatternRewriter rewriter,
      const MlirBeaverCompilerKernelOperation *operation,
      intptr_t nRegions, MlirOperation *created,
      MlirStringCallback diagnostic, void *diagnosticUserData);
  MlirLogicalResult (*replaceOperationWithRegions)(
      MlirConversionPatternRewriter rewriter, MlirOperation replacement,
      MlirOperation source, intptr_t expectedRegions,
      MlirStringCallback diagnostic, void *diagnosticUserData);
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
