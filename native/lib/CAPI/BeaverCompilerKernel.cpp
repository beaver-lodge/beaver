#include "mlir-c/Beaver/CompilerKernel.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
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
  void *userData;
};

static const MlirBeaverCompilerKernelHostAPI &hostAPI();

static void eraseOperation(MlirConversionPatternRewriter rewriter,
                           MlirOperation operation) {
  MlirPatternRewriter patternRewriter =
      mlirConversionPatternRewriterAsPatternRewriter(rewriter);
  mlirRewriterBaseEraseOp(mlirPatternRewriterAsBase(patternRewriter),
                          operation);
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
        &hostAPI(), operation, nOperands, operands, rewriter, state->userData,
        appendDiagnostic, &diagnostic);

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
        descriptor->userData});

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
      sizeof(MlirBeaverCompilerKernelHostAPI), addPattern, eraseOperation};
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
