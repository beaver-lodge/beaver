#include "mlir-c/Beaver/CompilerKernel.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

constexpr size_t maxErrorLength = 2048;
thread_local std::string lastCompilerKernelError;

struct PatternIdentity {
  std::string name;
  std::string root;
  std::string version;

  bool operator==(const PatternIdentity &other) const {
    return std::tie(name, root, version) ==
           std::tie(other.name, other.root, other.version);
  }
};

struct PopulationState {
  MlirRewritePatternSet patterns;
  MlirTypeConverter typeConverter;
  std::vector<PatternIdentity> actualPatterns;
  std::string error;
};

struct ExternalPatternState {
  PatternIdentity identity;
  MlirBeaverCompilerKernelRewriteFn rewrite;
  MlirBeaverCompilerKernelDestroyFn destroy;
  MlirTypeConverter typeConverter;
  void *userData;
};

static const MlirBeaverCompilerKernelHostAPI &hostAPI();
static bool copyStringRef(MlirStringRef value, std::string &output);

static MlirLogicalResult hostFailure(MlirStringCallback diagnostic,
                                     void *diagnosticUserData,
                                     llvm::StringRef message) {
  if (diagnostic)
    diagnostic(mlirStringRefCreate(message.data(), message.size()),
               diagnosticUserData);
  return mlirLogicalResultFailure();
}

static MlirRewriterBase rewriterBase(MlirConversionPatternRewriter rewriter) {
  MlirPatternRewriter patternRewriter =
      mlirConversionPatternRewriterAsPatternRewriter(rewriter);
  return mlirPatternRewriterAsBase(patternRewriter);
}

static bool sameContext(MlirContext left, MlirContext right) {
  return left.ptr && right.ptr && mlirContextEqual(left, right);
}

static MlirLogicalResult operationResult(
    MlirOperation operation, intptr_t index, MlirValue *result,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (!result || mlirOperationIsNull(operation) || index < 0 ||
      index >= mlirOperationGetNumResults(operation))
    return hostFailure(diagnostic, diagnosticUserData,
                       "operation result index is out of bounds");

  *result = mlirOperationGetResult(operation, index);
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult operationLocation(
    MlirOperation operation, MlirLocation *location,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (!location || mlirOperationIsNull(operation))
    return hostFailure(diagnostic, diagnosticUserData,
                       "cannot inspect a null operation");

  *location = mlirOperationGetLocation(operation);
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult valueType(MlirValue value, MlirType *type,
                                   MlirStringCallback diagnostic,
                                   void *diagnosticUserData) {
  if (!type || mlirValueIsNull(value))
    return hostFailure(diagnostic, diagnosticUserData,
                       "cannot inspect a null value");

  *type = mlirValueGetType(value);
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult convertType(MlirTypeConverter converter,
                                     MlirType type, MlirType *converted,
                                     MlirStringCallback diagnostic,
                                     void *diagnosticUserData) {
  if (!converter.ptr || mlirTypeIsNull(type) || !converted)
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid type conversion request");

  try {
    Type result = unwrap(converter)->convertType(unwrap(type));
    if (!result)
      return hostFailure(diagnostic, diagnosticUserData,
                         "type converter rejected the source type");

    *converted = wrap(result);
    return mlirLogicalResultSuccess();
  } catch (const std::exception &error) {
    return hostFailure(diagnostic, diagnosticUserData, error.what());
  } catch (...) {
    return hostFailure(diagnostic, diagnosticUserData,
                       "exception while converting a type");
  }
}

static bool validArray(intptr_t size, const void *data) {
  return size >= 0 && (size == 0 || data != nullptr);
}

static MlirLogicalResult createOperation(
    MlirConversionPatternRewriter rewriter,
    const MlirBeaverCompilerKernelOperation *descriptor,
    MlirOperation *created, MlirStringCallback diagnostic,
    void *diagnosticUserData) {
  if (!rewriter.ptr || !descriptor ||
      descriptor->structSize < sizeof(MlirBeaverCompilerKernelOperation) ||
      !created || mlirLocationIsNull(descriptor->location) ||
      !validArray(descriptor->nOperands, descriptor->operands) ||
      !validArray(descriptor->nResultTypes, descriptor->resultTypes) ||
      !validArray(descriptor->nAttributes, descriptor->attributes))
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid scalar operation descriptor");

  std::string name;
  if (!copyStringRef(descriptor->name, name) || name.empty())
    return hostFailure(diagnostic, diagnosticUserData,
                       "operation name must be a non-empty string");

  try {
    MlirRewriterBase base = rewriterBase(rewriter);
    MlirContext context = mlirRewriterBaseGetContext(base);
    if (!sameContext(context, mlirLocationGetContext(descriptor->location)))
      return hostFailure(diagnostic, diagnosticUserData,
                         "operation location belongs to a foreign context");

    for (intptr_t index = 0; index < descriptor->nOperands; ++index) {
      MlirValue operand = descriptor->operands[index];
      if (mlirValueIsNull(operand) ||
          !sameContext(context,
                       mlirTypeGetContext(mlirValueGetType(operand))))
        return hostFailure(diagnostic, diagnosticUserData,
                           "operation operand belongs to a foreign context");
    }

    for (intptr_t index = 0; index < descriptor->nResultTypes; ++index) {
      MlirType resultType = descriptor->resultTypes[index];
      if (mlirTypeIsNull(resultType) ||
          !sameContext(context, mlirTypeGetContext(resultType)))
        return hostFailure(diagnostic, diagnosticUserData,
                           "operation result type belongs to a foreign context");
    }

    for (intptr_t index = 0; index < descriptor->nAttributes; ++index) {
      MlirNamedAttribute attribute = descriptor->attributes[index];
      if (!attribute.name.ptr || mlirAttributeIsNull(attribute.attribute) ||
          !sameContext(context, mlirIdentifierGetContext(attribute.name)) ||
          !sameContext(context, mlirAttributeGetContext(attribute.attribute)))
        return hostFailure(diagnostic, diagnosticUserData,
                           "operation attribute belongs to a foreign context");
    }

    MlirOperationState state =
        mlirOperationStateGet(descriptor->name, descriptor->location);
    mlirOperationStateAddOperands(&state, descriptor->nOperands,
                                  descriptor->operands);
    mlirOperationStateAddResults(&state, descriptor->nResultTypes,
                                 descriptor->resultTypes);
    mlirOperationStateAddAttributes(&state, descriptor->nAttributes,
                                    descriptor->attributes);

    MlirOperation operation = mlirOperationCreate(&state);
    if (mlirOperationIsNull(operation))
      return hostFailure(diagnostic, diagnosticUserData,
                         "MLIR rejected scalar operation construction");

    *created = mlirRewriterBaseInsert(base, operation);
    return mlirLogicalResultSuccess();
  } catch (const std::exception &error) {
    return hostFailure(diagnostic, diagnosticUserData, error.what());
  } catch (...) {
    return hostFailure(diagnostic, diagnosticUserData,
                       "exception while creating a scalar operation");
  }
}

static MlirLogicalResult replaceOperationWithValues(
    MlirConversionPatternRewriter rewriter, MlirOperation operation,
    intptr_t nValues, const MlirValue *values,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (!rewriter.ptr || mlirOperationIsNull(operation) ||
      !validArray(nValues, values) ||
      nValues != mlirOperationGetNumResults(operation))
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid operation replacement request");

  MlirRewriterBase base = rewriterBase(rewriter);
  MlirContext context = mlirRewriterBaseGetContext(base);
  if (!sameContext(context, mlirOperationGetContext(operation)))
    return hostFailure(diagnostic, diagnosticUserData,
                       "operation belongs to a foreign context");

  for (intptr_t index = 0; index < nValues; ++index) {
    if (mlirValueIsNull(values[index]) ||
        !sameContext(context,
                     mlirTypeGetContext(mlirValueGetType(values[index]))))
      return hostFailure(diagnostic, diagnosticUserData,
                         "replacement value belongs to a foreign context");
  }

  mlirRewriterBaseReplaceOpWithValues(base, operation, nValues, values);
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult eraseOperation(
    MlirConversionPatternRewriter rewriter, MlirOperation operation,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (!rewriter.ptr || mlirOperationIsNull(operation))
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid operation erase request");

  MlirRewriterBase base = rewriterBase(rewriter);
  if (!sameContext(mlirRewriterBaseGetContext(base),
                   mlirOperationGetContext(operation)))
    return hostFailure(diagnostic, diagnosticUserData,
                       "operation belongs to a foreign context");

  mlirRewriterBaseEraseOp(base, operation);
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult operationAttribute(
    MlirOperation operation, MlirStringRef name, MlirAttribute *attribute,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  std::string key;
  if (!attribute || mlirOperationIsNull(operation) ||
      !copyStringRef(name, key) || key.empty())
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid operation attribute request");

  *attribute = mlirOperationGetAttributeByName(operation, name);
  if (mlirAttributeIsNull(*attribute))
    return hostFailure(diagnostic, diagnosticUserData,
                       "operation attribute is missing");

  return mlirLogicalResultSuccess();
}

static MlirLogicalResult attributeStringValue(
    MlirAttribute attribute, MlirStringRef *value,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (!value || mlirAttributeIsNull(attribute) ||
      !mlirAttributeIsAString(attribute))
    return hostFailure(diagnostic, diagnosticUserData,
                       "attribute is not a string");

  *value = mlirStringAttrGetValue(attribute);
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult integerType(
    MlirConversionPatternRewriter rewriter, unsigned width, MlirType *type,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (!rewriter.ptr || !type || width == 0 || width > 4096)
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid integer type request");

  MlirContext context = mlirRewriterBaseGetContext(rewriterBase(rewriter));
  *type = mlirIntegerTypeGet(context, width);
  if (mlirTypeIsNull(*type))
    return hostFailure(diagnostic, diagnosticUserData,
                       "failed to create integer type");

  return mlirLogicalResultSuccess();
}

static MlirLogicalResult integerAttribute(
    MlirType type, int64_t value, MlirAttribute *attribute,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (!attribute || mlirTypeIsNull(type) || !mlirTypeIsAInteger(type))
    return hostFailure(diagnostic, diagnosticUserData,
                       "integer attribute requires an integer type");

  *attribute = mlirIntegerAttrGet(type, value);
  if (mlirAttributeIsNull(*attribute))
    return hostFailure(diagnostic, diagnosticUserData,
                       "failed to create integer attribute");

  return mlirLogicalResultSuccess();
}

static MlirLogicalResult namedAttribute(
    MlirConversionPatternRewriter rewriter, MlirStringRef name,
    MlirAttribute attribute, MlirNamedAttribute *namedAttribute,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  std::string key;
  if (!rewriter.ptr || !namedAttribute || mlirAttributeIsNull(attribute) ||
      !copyStringRef(name, key) || key.empty())
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid named attribute request");

  MlirContext context = mlirRewriterBaseGetContext(rewriterBase(rewriter));
  if (!sameContext(context, mlirAttributeGetContext(attribute)))
    return hostFailure(diagnostic, diagnosticUserData,
                       "attribute belongs to a foreign context");

  MlirIdentifier identifier = mlirIdentifierGet(context, name);
  if (!identifier.ptr)
    return hostFailure(diagnostic, diagnosticUserData,
                       "failed to create attribute identifier");

  *namedAttribute = mlirNamedAttributeGet(identifier, attribute);
  return mlirLogicalResultSuccess();
}

static MlirLogicalResult operationCounts(
    MlirOperation operation, intptr_t *nOperands, intptr_t *nResults,
    MlirStringCallback diagnostic, void *diagnosticUserData) {
  if (mlirOperationIsNull(operation) || !nOperands || !nResults)
    return hostFailure(diagnostic, diagnosticUserData,
                       "invalid operation count request");

  *nOperands = mlirOperationGetNumOperands(operation);
  *nResults = mlirOperationGetNumResults(operation);
  return mlirLogicalResultSuccess();
}

static MlirStringRef stringRef(const std::string &value) {
  return mlirStringRefCreate(value.data(), value.size());
}

static MlirStringRef succeed() {
  lastCompilerKernelError.clear();
  return stringRef(lastCompilerKernelError);
}

static MlirStringRef fail(llvm::StringRef code, llvm::StringRef message) {
  lastCompilerKernelError = (code + "|" + message).str();

  if (lastCompilerKernelError.size() > maxErrorLength)
    lastCompilerKernelError.resize(maxErrorLength);

  return stringRef(lastCompilerKernelError);
}

static bool copyStringRef(MlirStringRef value, std::string &output) {
  if (value.length != 0 && value.data == nullptr)
    return false;

  output.assign(value.data == nullptr ? "" : value.data, value.length);
  return output.find('\0') == std::string::npos;
}

static void appendDiagnostic(MlirStringRef chunk, void *userData) {
  auto *message = static_cast<std::string *>(userData);
  if (!message || !chunk.data || chunk.length == 0 ||
      message->size() >= maxErrorLength)
    return;

  size_t remaining = maxErrorLength - message->size();
  message->append(chunk.data, std::min(chunk.length, remaining));
}

static void setPopulationError(PopulationState &state, llvm::StringRef error) {
  if (state.error.empty())
    state.error = error.take_front(maxErrorLength).str();
}

static MlirLogicalResult externalPatternRewrite(
    MlirConversionPattern, MlirOperation operation, intptr_t nOperands,
    MlirValue *operands, MlirConversionPatternRewriter rewriter,
    void *userData) {
  auto *state = static_cast<ExternalPatternState *>(userData);

  if (!state || !state->rewrite)
    return mlirLogicalResultFailure();

  std::string diagnostic;

  try {
    MlirLogicalResult result = state->rewrite(
        &hostAPI(), operation, nOperands, operands, rewriter,
        state->typeConverter, state->userData, appendDiagnostic, &diagnostic);

    if (mlirLogicalResultIsFailure(result) && !diagnostic.empty())
      unwrap(operation)->emitError() << diagnostic;

    return result;
  } catch (const std::exception &error) {
    unwrap(operation)->emitError()
        << "external compiler-kernel pattern '" << state->identity.name
        << "' threw across its C ABI boundary: " << error.what();
  } catch (...) {
    unwrap(operation)->emitError()
        << "external compiler-kernel pattern '" << state->identity.name
        << "' threw across its C ABI boundary";
  }

  return mlirLogicalResultFailure();
}

static void destroyExternalPattern(void *userData) {
  auto *state = static_cast<ExternalPatternState *>(userData);
  if (!state)
    return;

  try {
    if (state->destroy)
      state->destroy(state->userData);
  } catch (...) {
    // Destructors cannot report through MLIR and must never unwind across the
    // C ABI. The pattern-owned state is still released below.
  }

  delete state;
}

static MlirLogicalResult addPattern(
    void *hostContext, MlirRewritePatternSet patterns,
    MlirTypeConverter typeConverter,
    const MlirBeaverCompilerKernelPattern *descriptor) {
  auto *population = static_cast<PopulationState *>(hostContext);
  bool recordedIdentity = false;

  try {
    if (!population || patterns.ptr != population->patterns.ptr ||
        typeConverter.ptr != population->typeConverter.ptr) {
      if (population)
        setPopulationError(*population, "foreign pattern-set or type-converter handle");
      return mlirLogicalResultFailure();
    }

    if (!descriptor ||
        descriptor->structSize < sizeof(MlirBeaverCompilerKernelPattern) ||
        !descriptor->matchAndRewrite) {
      setPopulationError(*population, "invalid external pattern descriptor");
      return mlirLogicalResultFailure();
    }

    PatternIdentity identity;
    if (!copyStringRef(descriptor->name, identity.name) ||
        !copyStringRef(descriptor->root, identity.root) ||
        !copyStringRef(descriptor->version, identity.version) ||
        identity.name.empty() || identity.root.empty() || identity.version.empty()) {
      setPopulationError(*population, "invalid external pattern identity");
      return mlirLogicalResultFailure();
    }

    auto duplicate = std::find_if(
        population->actualPatterns.begin(), population->actualPatterns.end(),
        [&](const PatternIdentity &existing) {
          return existing.name == identity.name || existing.root == identity.root;
        });

    if (duplicate != population->actualPatterns.end()) {
      setPopulationError(*population, "duplicate external pattern name or root");
      return mlirLogicalResultFailure();
    }

    population->actualPatterns.push_back(identity);
    recordedIdentity = true;

    auto state = std::make_unique<ExternalPatternState>(ExternalPatternState{
        identity, descriptor->matchAndRewrite, descriptor->destroy,
        typeConverter, descriptor->userData});

    MlirConversionPatternCallbacks callbacks = {
        nullptr, destroyExternalPattern, externalPatternRewrite, nullptr};
    MlirContext context = wrap(unwrap(patterns)->getContext());
    MlirConversionPattern pattern = mlirOpConversionPatternCreate(
        descriptor->root, descriptor->benefit, context, typeConverter,
        callbacks, state.get(), 0, nullptr);
    mlirRewritePatternSetAdd(patterns,
                             mlirConversionPatternAsRewritePattern(pattern));
    state.release();
    return mlirLogicalResultSuccess();
  } catch (const std::exception &error) {
    if (population && recordedIdentity)
      population->actualPatterns.pop_back();
    if (population)
      setPopulationError(*population, error.what());
  } catch (...) {
    if (population && recordedIdentity)
      population->actualPatterns.pop_back();
    if (population)
      setPopulationError(*population, "exception while registering external pattern");
  }

  return mlirLogicalResultFailure();
}

static const MlirBeaverCompilerKernelHostAPI &hostAPI() {
  static const MlirBeaverCompilerKernelHostAPI api = {
      MLIR_BEAVER_COMPILER_KERNEL_ABI_VERSION,
      sizeof(MlirBeaverCompilerKernelHostAPI),
      addPattern,
      operationResult,
      operationLocation,
      valueType,
      convertType,
      createOperation,
      replaceOperationWithValues,
      eraseOperation,
      operationAttribute,
      attributeStringValue,
      integerType,
      integerAttribute,
      namedAttribute,
      operationCounts};
  return api;
}

static llvm::Expected<std::vector<PatternIdentity>>
parseExpectedPatterns(llvm::StringRef json) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return parsed.takeError();

  auto *array = parsed->getAsArray();
  if (!array)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "expected pattern manifest array");

  std::vector<PatternIdentity> patterns;
  patterns.reserve(array->size());

  for (const llvm::json::Value &value : *array) {
    auto *object = value.getAsObject();
    if (!object)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "expected pattern manifest object");

    auto name = object->getString("name");
    auto root = object->getString("root");
    auto version = object->getString("version");
    if (!name || !root || !version)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "expected pattern manifest name, root, and version strings");

    patterns.push_back({name->str(), root->str(), version->str()});
  }

  return patterns;
}

template <typename Function>
static Function symbol(llvm::sys::DynamicLibrary &library,
                       const std::string &name) {
  return reinterpret_cast<Function>(library.getAddressOfSymbol(name.c_str()));
}

} // namespace

MlirStringRef beaverCompilerKernelLoadAndPopulate(
    MlirRewritePatternSet patterns, MlirTypeConverter typeConverter,
    MlirStringRef artifactPath, MlirStringRef abiVersionSymbol,
    MlirStringRef manifestSymbol, MlirStringRef populateSymbol,
    MlirStringRef expectedIdentity, MlirStringRef expectedPatternsJSON) {
  try {
    std::string path;
    std::string abiName;
    std::string manifestName;
    std::string populateName;
    std::string identity;
    std::string patternsJSON;

    if (!patterns.ptr || !typeConverter.ptr ||
        !copyStringRef(artifactPath, path) || path.empty() ||
        !copyStringRef(abiVersionSymbol, abiName) || abiName.empty() ||
        !copyStringRef(manifestSymbol, manifestName) || manifestName.empty() ||
        !copyStringRef(populateSymbol, populateName) || populateName.empty() ||
        !copyStringRef(expectedIdentity, identity) || identity.empty() ||
        !copyStringRef(expectedPatternsJSON, patternsJSON))
      return fail("E_INVALID_ARGUMENT", "invalid loader argument");

    auto expectedPatterns = parseExpectedPatterns(patternsJSON);
    if (!expectedPatterns)
      return fail("E_PATTERN_MANIFEST_INVALID",
                  llvm::toString(expectedPatterns.takeError()));

    std::string loadError;
    llvm::sys::DynamicLibrary library =
        llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &loadError);
    if (!library.isValid())
      return fail("E_DLOPEN", loadError);

    auto abi = symbol<MlirBeaverCompilerKernelABIVersionFn>(library, abiName);
    auto embeddedManifest =
        symbol<MlirBeaverCompilerKernelManifestFn>(library, manifestName);
    auto populate =
        symbol<MlirBeaverCompilerKernelPopulateFn>(library, populateName);

    if (!abi)
      return fail("E_MISSING_SYMBOL", abiName);
    if (!embeddedManifest)
      return fail("E_MISSING_SYMBOL", manifestName);
    if (!populate)
      return fail("E_MISSING_SYMBOL", populateName);

    if (abi() != MLIR_BEAVER_COMPILER_KERNEL_ABI_VERSION)
      return fail("E_ABI_MISMATCH", "artifact returned an unsupported ABI version");

    std::string embeddedIdentity;
    if (!copyStringRef(embeddedManifest(), embeddedIdentity) ||
        embeddedIdentity != identity)
      return fail("E_MANIFEST_IDENTITY_MISMATCH",
                  "artifact identity does not match its sidecar manifest");

    PopulationState population{patterns, typeConverter, {}, {}};
    MlirLogicalResult populated =
        populate(patterns, typeConverter, &hostAPI(), &population,
                 appendDiagnostic, &population.error);

    if (mlirLogicalResultIsFailure(populated) || !population.error.empty())
      return fail("E_POPULATE_FAILED",
                  population.error.empty() ? "artifact rejected population"
                                           : population.error);

    if (population.actualPatterns != *expectedPatterns)
      return fail("E_PATTERN_MANIFEST_MISMATCH",
                  "registered patterns differ from the sidecar manifest");

    return succeed();
  } catch (const std::exception &error) {
    return fail("E_EXCEPTION", error.what());
  } catch (...) {
    return fail("E_EXCEPTION", "exception crossed the compiler-kernel ABI");
  }
}
