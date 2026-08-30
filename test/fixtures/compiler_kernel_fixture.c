#include "mlir-c/Beaver/CompilerKernel.h"

#ifndef FIXTURE_ABI_VERSION
#define FIXTURE_ABI_VERSION MLIR_BEAVER_COMPILER_KERNEL_ABI_VERSION
#endif

#if defined(FIXTURE_SYMBOL_PATTERN)
static MlirLogicalResult fixture_symbol_rewrite(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, MlirTypeConverter typeConverter,
    void *userData, MlirStringCallback diagnostic,
    void *diagnosticUserData) {
  (void)userData;
  if (nOperands != 2)
    return mlirLogicalResultFailure();

  MlirValue sourceResult;
  MlirType sourceResultType;
  MlirType resultType;
  MlirType inputTypes[2];
  MlirLocation location;
  MlirAttribute callee;
  MlirNamedAttribute namedCallee;

  if (mlirLogicalResultIsFailure(host->operationResult(
          operation, 0, &sourceResult, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->valueType(
          sourceResult, &sourceResultType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->convertType(
          typeConverter, sourceResultType, &resultType, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->valueType(
          operands[0], &inputTypes[0], diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->valueType(
          operands[1], &inputTypes[1], diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationLocation(
          operation, &location, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->ensureFunctionDeclaration(
          operation, rewriter, mlirStringRefCreate("fixture.runtime", 15), 2,
          inputTypes, 1, &resultType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->flatSymbolRefAttribute(
          rewriter, mlirStringRefCreate("fixture.runtime", 15), &callee,
          diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->namedAttribute(
          rewriter, mlirStringRefCreate("callee", 6), callee, &namedCallee,
          diagnostic, diagnosticUserData)))
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelOperation descriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("func.call", sizeof("func.call") - 1),
      location,
      nOperands,
      operands,
      1,
      &resultType,
      1,
      &namedCallee,
  };

  MlirOperation replacement;
  MlirValue replacementResult;
  if (mlirLogicalResultIsFailure(host->createOperation(
          rewriter, &descriptor, &replacement, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationResult(
          replacement, 0, &replacementResult, diagnostic,
          diagnosticUserData)))
    return mlirLogicalResultFailure();

  return host->replaceOperationWithValues(
      rewriter, operation, 1, &replacementResult, diagnostic,
      diagnosticUserData);
}
#endif

#if defined(FIXTURE_REGION_PATTERN)
static MlirLogicalResult fixture_region_rewrite(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, MlirTypeConverter typeConverter,
    void *userData, MlirStringCallback diagnostic,
    void *diagnosticUserData) {
  (void)operands;
  (void)userData;
  if (nOperands != 0)
    return mlirLogicalResultFailure();

  MlirAttribute arityAttribute;
  int64_t arity;
  MlirBlock block;
  intptr_t blockArguments;
  MlirOperation terminator;
  MlirValue returnValue;
  MlirType returnValueType;
  MlirType outputType;
  MlirType signature;
  MlirAttribute signatureAttribute;
  MlirNamedAttribute namedSignature;
  MlirValue sourceResult;
  MlirType sourceResultType;
  MlirType resultType;
  MlirLocation location;
  MlirType pointerType;
  const int32_t segments[2] = {2, 0};
  MlirAttribute segmentsAttribute;
  MlirNamedAttribute namedSegments;
  intptr_t regionCount;

  if (mlirLogicalResultIsFailure(host->operationAttribute(
          operation, mlirStringRefCreate("arity", 5), &arityAttribute,
          diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationRegionCount(
          operation, &regionCount, diagnostic, diagnosticUserData)) ||
      regionCount != 1 ||
      mlirLogicalResultIsFailure(host->attributeIntegerValue(
          arityAttribute, &arity, diagnostic, diagnosticUserData)) ||
      arity != 0 ||
      mlirLogicalResultIsFailure(host->singleRegionBlock(
          operation, 0, &block, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->blockArgumentCount(
          block, &blockArguments, diagnostic, diagnosticUserData)) ||
      blockArguments != 0 ||
      mlirLogicalResultIsFailure(host->blockTerminator(
          block, &terminator, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationOperand(
          terminator, 0, &returnValue, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->valueType(
          returnValue, &returnValueType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->convertType(
          typeConverter, returnValueType, &outputType, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->functionType(
          rewriter, 0, NULL, 1, &outputType, &signature, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->typeAttribute(
          signature, &signatureAttribute, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->namedAttribute(
          rewriter, mlirStringRefCreate("signature", 9), signatureAttribute,
          &namedSignature, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationResult(
          operation, 0, &sourceResult, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->valueType(
          sourceResult, &sourceResultType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->convertType(
          typeConverter, sourceResultType, &resultType, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationLocation(
          operation, &location, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->llvmPointerType(
          rewriter, 0, &pointerType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->denseI32ArrayAttribute(
#if defined(FIXTURE_CONTROL_BAD_DENSE)
          rewriter, -1,
#else
          rewriter, 2,
#endif
          segments, &segmentsAttribute, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->namedAttribute(
          rewriter, mlirStringRefCreate("segments", 8), segmentsAttribute,
          &namedSegments, diagnostic, diagnosticUserData)))
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelOperation startDescriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("fixture.control_start",
                          sizeof("fixture.control_start") - 1),
      location,
      0,
      NULL,
      1,
      &pointerType,
      1,
      &namedSegments,
  };
  MlirBeaverCompilerKernelOperation endDescriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("fixture.control_end",
                          sizeof("fixture.control_end") - 1),
      location,
      0,
      NULL,
      0,
      NULL,
      0,
      NULL,
  };
  MlirOperation start;
  MlirOperation end;
  if (mlirLogicalResultIsFailure(host->createOperationAtBlockStart(
          rewriter, block, &startDescriptor, &start, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->createOperationBefore(
          rewriter, terminator, &endDescriptor, &end, diagnostic,
          diagnosticUserData)))
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelOperation descriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("fixture.lowered", sizeof("fixture.lowered") - 1),
      location,
      0,
      NULL,
      1,
      &resultType,
      1,
      &namedSignature,
  };

  MlirOperation replacement;
#if defined(FIXTURE_REGION_MISMATCH)
  const intptr_t replacementRegions = 2;
#else
  const intptr_t replacementRegions = 1;
#endif
  if (mlirLogicalResultIsFailure(host->createOperationWithRegions(
          rewriter, &descriptor, replacementRegions, &replacement, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->replaceOperationWithRegions(
          rewriter, replacement, operation, 1, diagnostic,
          diagnosticUserData)))
    return mlirLogicalResultFailure();

  return mlirLogicalResultSuccess();
}
#endif

#if defined(FIXTURE_TYPE_PATTERN)
static MlirLogicalResult fixture_type_rewrite(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, MlirTypeConverter typeConverter,
    void *userData, MlirStringCallback diagnostic,
    void *diagnosticUserData) {
  (void)typeConverter;
  (void)userData;
  if (nOperands != 1)
    return mlirLogicalResultFailure();

  MlirType operandType;
  int isI64;
  MlirStringRef dynamicName;
  MlirType resultType;
  MlirAttribute value;
  MlirNamedAttribute namedValue;
  MlirLocation location;

#if defined(FIXTURE_TYPE_BAD_WIDTH)
  const unsigned width = 0;
#else
  const unsigned width = 64;
#endif

  if (mlirLogicalResultIsFailure(host->valueType(
          operands[0], &operandType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->typeIsInteger(
          operandType, width, &isI64, diagnostic, diagnosticUserData)) ||
      (!isI64 &&
       (mlirLogicalResultIsFailure(host->dynamicTypeName(
            operandType, &dynamicName, diagnostic, diagnosticUserData)) ||
        dynamicName.length != 4 || dynamicName.data[0] != 't' ||
        dynamicName.data[1] != 'e' || dynamicName.data[2] != 'r' ||
        dynamicName.data[3] != 'm')) ||
      mlirLogicalResultIsFailure(host->integerType(
          rewriter, 64, &resultType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->integerAttribute(
          resultType, isI64, &value, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->namedAttribute(
          rewriter, mlirStringRefCreate("value", 5), value, &namedValue,
          diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationLocation(
          operation, &location, diagnostic, diagnosticUserData)))
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelOperation descriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("arith.constant", sizeof("arith.constant") - 1),
      location,
      0,
      NULL,
      1,
      &resultType,
      1,
      &namedValue,
  };

  MlirOperation replacement;
  MlirValue replacementResult;
  if (mlirLogicalResultIsFailure(host->createOperation(
          rewriter, &descriptor, &replacement, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationResult(
          replacement, 0, &replacementResult, diagnostic,
          diagnosticUserData)))
    return mlirLogicalResultFailure();

  return host->replaceOperationWithValues(
      rewriter, operation, 1, &replacementResult, diagnostic,
      diagnosticUserData);
}
#endif

#ifndef FIXTURE_IDENTITY
#error "FIXTURE_IDENTITY must be defined"
#endif

#if defined(_WIN32)
#define FIXTURE_EXPORT __declspec(dllexport)
#else
#define FIXTURE_EXPORT __attribute__((visibility("default")))
#endif

static MlirLogicalResult fixture_rewrite(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, MlirTypeConverter typeConverter,
    void *userData,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  (void)host;
  (void)operation;
  (void)userData;

  if (nOperands != 2) {
    static const char message[] = "fixture.add expects two converted operands";
    diagnostic(mlirStringRefCreate(message, sizeof(message) - 1),
               diagnosticUserData);
    return mlirLogicalResultFailure();
  }

  MlirValue sourceResult;
  MlirType sourceType;
  MlirType convertedType;
  MlirLocation location;
#if defined(FIXTURE_BAD_RESULT_INDEX)
  const intptr_t sourceResultIndex = 1;
#else
  const intptr_t sourceResultIndex = 0;
#endif
  if (mlirLogicalResultIsFailure(host->operationResult(
          operation, sourceResultIndex, &sourceResult, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationLocation(
          operation, &location, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->valueType(
          sourceResult, &sourceType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->convertType(
          typeConverter, sourceType, &convertedType,
          diagnostic, diagnosticUserData)))
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelOperation descriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("arith.addi", sizeof("arith.addi") - 1),
      location,
      nOperands,
      operands,
      1,
      &convertedType,
      0,
      NULL,
  };

  MlirOperation replacement;
  MlirValue replacementResult;
  if (mlirLogicalResultIsFailure(host->createOperation(
          rewriter, &descriptor, &replacement, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationResult(
          replacement, 0, &replacementResult, diagnostic,
          diagnosticUserData)))
    return mlirLogicalResultFailure();

#if defined(FIXTURE_PARTIAL_FAILURE)
  static const char message[] = "fixture requested rollback after operation creation";
  diagnostic(mlirStringRefCreate(message, sizeof(message) - 1),
             diagnosticUserData);
  return mlirLogicalResultFailure();
#endif

  return host->replaceOperationWithValues(
      rewriter, operation, 1, &replacementResult, diagnostic,
      diagnosticUserData);
}

#if defined(FIXTURE_ATTRIBUTE_PATTERN)
static MlirLogicalResult fixture_attribute_rewrite(
    const MlirBeaverCompilerKernelHostAPI *host, MlirOperation operation,
    intptr_t nOperands, MlirValue *operands,
    MlirConversionPatternRewriter rewriter, MlirTypeConverter typeConverter,
    void *userData, MlirStringCallback diagnostic,
    void *diagnosticUserData) {
  (void)operands;
  (void)typeConverter;
  (void)userData;

  if (nOperands != 0)
    return mlirLogicalResultFailure();

  MlirAttribute predicate;
  MlirStringRef predicateValue;
  MlirType resultType;
  MlirAttribute value;
  MlirNamedAttribute namedValue;
  MlirLocation location;
  MlirValue sourceResult;
  intptr_t sourceOperands;
  intptr_t sourceResults;

  if (mlirLogicalResultIsFailure(host->operationCounts(
          operation, &sourceOperands, &sourceResults, diagnostic,
          diagnosticUserData)) ||
      sourceOperands != 0 || sourceResults != 1 ||
      mlirLogicalResultIsFailure(host->operationResult(
          operation, 0, &sourceResult, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationLocation(
          operation, &location, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationAttribute(
          operation, mlirStringRefCreate("predicate", 9), &predicate,
          diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->attributeStringValue(
          predicate, &predicateValue, diagnostic, diagnosticUserData)) ||
      predicateValue.length != 2 || predicateValue.data[0] != 'e' ||
      predicateValue.data[1] != 'q' ||
      mlirLogicalResultIsFailure(host->integerType(
          rewriter, 64, &resultType, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->integerAttribute(
          resultType, 42, &value, diagnostic, diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->namedAttribute(
          rewriter, mlirStringRefCreate("value", 5), value, &namedValue,
          diagnostic, diagnosticUserData)))
    return mlirLogicalResultFailure();

  MlirBeaverCompilerKernelOperation descriptor = {
      sizeof(MlirBeaverCompilerKernelOperation),
      mlirStringRefCreate("arith.constant", sizeof("arith.constant") - 1),
      location,
      0,
      NULL,
      1,
      &resultType,
      1,
      &namedValue,
  };

  MlirOperation replacement;
  MlirValue replacementResult;
  if (mlirLogicalResultIsFailure(host->createOperation(
          rewriter, &descriptor, &replacement, diagnostic,
          diagnosticUserData)) ||
      mlirLogicalResultIsFailure(host->operationResult(
          replacement, 0, &replacementResult, diagnostic,
          diagnosticUserData)))
    return mlirLogicalResultFailure();

  return host->replaceOperationWithValues(
      rewriter, operation, 1, &replacementResult, diagnostic,
      diagnosticUserData);
}
#endif

FIXTURE_EXPORT uint32_t fixture_abi_version(void) {
  return FIXTURE_ABI_VERSION;
}

FIXTURE_EXPORT MlirStringRef fixture_manifest(void) {
  static const char identity[] = FIXTURE_IDENTITY;
  return mlirStringRefCreate(identity, sizeof(identity) - 1);
}

FIXTURE_EXPORT MlirLogicalResult fixture_populate(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter,
    const MlirBeaverCompilerKernelHostAPI *host, void *hostContext,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  (void)diagnostic;
  (void)diagnosticUserData;

  if (!host || host->abiVersion != MLIR_BEAVER_COMPILER_KERNEL_ABI_VERSION ||
      host->structSize < sizeof(MlirBeaverCompilerKernelHostAPI) ||
      !host->addPattern || !host->operationResult ||
      !host->operationLocation || !host->valueType || !host->convertType ||
      !host->createOperation || !host->replaceOperationWithValues ||
      !host->eraseOperation)
    return mlirLogicalResultFailure();

#if defined(FIXTURE_ATTRIBUTE_PATTERN)
  if (!host->operationAttribute || !host->attributeStringValue ||
      !host->integerType || !host->integerAttribute || !host->namedAttribute ||
      !host->operationCounts)
    return mlirLogicalResultFailure();
#endif

#if defined(FIXTURE_SYMBOL_PATTERN)
  if (!host->flatSymbolRefAttribute || !host->ensureFunctionDeclaration ||
      !host->namedAttribute)
    return mlirLogicalResultFailure();
#endif

#if defined(FIXTURE_REGION_PATTERN)
  if (!host->operationOperand || !host->attributeIntegerValue ||
      !host->singleRegionBlock || !host->blockArgumentCount ||
      !host->blockArgument ||
      !host->blockTerminator || !host->functionType || !host->typeAttribute ||
      !host->createOperationWithRegions ||
      !host->replaceOperationWithRegions ||
      !host->operationAttribute || !host->valueType || !host->convertType ||
      !host->operationResult || !host->operationLocation ||
      !host->namedAttribute || !host->llvmPointerType ||
      !host->denseI32ArrayAttribute || !host->createOperationAtBlockStart ||
      !host->createOperationBefore || !host->operationRegionCount)
    return mlirLogicalResultFailure();
#endif

#if defined(FIXTURE_TYPE_PATTERN)
  if (!host->typeIsInteger || !host->dynamicTypeName || !host->valueType ||
      !host->integerType || !host->integerAttribute || !host->namedAttribute ||
      !host->operationLocation || !host->createOperation ||
      !host->operationResult || !host->replaceOperationWithValues)
    return mlirLogicalResultFailure();
#endif

#if defined(FIXTURE_ATTRIBUTE_PATTERN)
  MlirBeaverCompilerKernelPattern pattern = {
      sizeof(MlirBeaverCompilerKernelPattern),
      mlirStringRefCreate("fixture.attr", sizeof("fixture.attr") - 1),
      mlirStringRefCreate("fixture.attr", sizeof("fixture.attr") - 1),
      mlirStringRefCreate("1", sizeof("1") - 1),
      1,
      fixture_attribute_rewrite,
      NULL,
      NULL,
  };
#elif defined(FIXTURE_SYMBOL_PATTERN)
  MlirBeaverCompilerKernelPattern pattern = {
      sizeof(MlirBeaverCompilerKernelPattern),
      mlirStringRefCreate("fixture.call", sizeof("fixture.call") - 1),
      mlirStringRefCreate("fixture.call", sizeof("fixture.call") - 1),
      mlirStringRefCreate("1", sizeof("1") - 1),
      1,
      fixture_symbol_rewrite,
      NULL,
      NULL,
  };
#elif defined(FIXTURE_REGION_PATTERN)
  MlirBeaverCompilerKernelPattern pattern = {
      sizeof(MlirBeaverCompilerKernelPattern),
      mlirStringRefCreate("fixture.region", sizeof("fixture.region") - 1),
      mlirStringRefCreate("scf.execute_region",
                          sizeof("scf.execute_region") - 1),
      mlirStringRefCreate("1", sizeof("1") - 1),
      1,
      fixture_region_rewrite,
      NULL,
      NULL,
  };
#elif defined(FIXTURE_TYPE_PATTERN)
  MlirBeaverCompilerKernelPattern pattern = {
      sizeof(MlirBeaverCompilerKernelPattern),
      mlirStringRefCreate("fixture.type", sizeof("fixture.type") - 1),
      mlirStringRefCreate("fixture.type", sizeof("fixture.type") - 1),
      mlirStringRefCreate("1", sizeof("1") - 1),
      1,
      fixture_type_rewrite,
      NULL,
      NULL,
  };
#else
  MlirBeaverCompilerKernelPattern pattern = {
      sizeof(MlirBeaverCompilerKernelPattern),
      mlirStringRefCreate("fixture.add", sizeof("fixture.add") - 1),
      mlirStringRefCreate("fixture.add", sizeof("fixture.add") - 1),
      mlirStringRefCreate("1", sizeof("1") - 1),
      1,
      fixture_rewrite,
      NULL,
      NULL,
  };
#endif

  return host->addPattern(hostContext, patterns, typeConverter, &pattern);
}
