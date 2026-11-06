#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Builder.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TableGen/Parser.h"
#include <map>

using namespace mlir;
namespace parser {
using llvm::json::Array;
using llvm::json::Object;
// TableGen already owns the complete schema. Serialize it directly rather than
// copying through a second, partial PDLL schema plus a parallel extras map.
static std::map<std::string, std::map<std::string, Object>> dialects;

static std::string processAndFormatDoc(const Twine &doc) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  raw_indented_ostream(stream).printReindented(StringRef(doc.str()).rtrim(" \t"));
  return text;
}

static Object constraint(StringRef name, const tblgen::Constraint &value, StringRef kind) {
  std::string unique = value.getUniqueDefName();
  StringRef display = StringRef(unique).split("(anonymous_").first.split("__beaver_ods__").first;
  return Object{{"name", name.str()}, {"constraint", display.empty() ? std::string("anonymous_constraint") : display.str()},
          {"description", value.getSummary().str()}, {"kind", kind}};
}

template <typename Range> static Array typeConstraints(Range values) {
  Array result;
  for (const auto &value : values)
    result.push_back(constraint(value.name, value.constraint,
                               value.isOptional() ? "Optional" : value.isVariadic() ? "Variadic" : "Single"));
  return result;
}

static void putArray(Object &object, StringRef key, Array values) {
  if (!values.empty()) object[key] = std::move(values);
}

void processTdIncludeRecords(const llvm::RecordKeeper &records) {
  for (const llvm::Record *def : records.getAllDerivedDefinitions("Op")) {
    tblgen::Operator op(def);
    auto &operations = dialects[op.getDialectName().str()];
    std::string name = op.getOperationName();
    if (operations.count(name)) continue;
    Object data{{"name", name}, {"summary", op.getSummary().str()},
                {"description", processAndFormatDoc(op.getDescription())},
                {"result_type_inference", bool(op.getTrait("::mlir::InferTypeOpInterface::Trait"))},
                {"native_class_name", op.getQualCppClassName()}, {"has_folder", op.hasFolder()}};
    Array traits, regions, builders, attributes, properties;
    for (const auto &trait : op.getTraits()) {
      if (const auto *native = llvm::dyn_cast<tblgen::NativeTrait>(&trait))
        traits.push_back(native->getFullyQualifiedTraitName());
      else if (StringRef traitName = trait.getDef().getName(); !traitName.starts_with("anonymous_"))
        traits.push_back(traitName.str());
    }
    for (const auto &region : op.getRegions())
      regions.push_back(Object{{"name", region.name.str()}, {"variadic", region.isVariadic()}});
    for (const auto &builder : op.getBuilders()) {
      Array params;
      for (const auto &param : builder.getParameters())
        params.push_back(Object{{"name", param.getName().value_or("").str()}, {"cpp_type", param.getCppType().str()},
                                {"has_default", param.getDefaultValue().has_value()}});
      builders.push_back(Object{{"deprecated", builder.getDeprecatedMessage().has_value()},
                                {"parameters", std::move(params)}});
    }
    for (const auto &attr : op.getAttributes())
      attributes.push_back(constraint(attr.name, attr.attr, attr.attr.isOptional() ? "Optional" : "Single"));
    for (const auto &property : op.getProperties()) {
      const auto &prop = property.prop;
      properties.push_back(Object{{"name", property.name.str()}, {"constraint", prop.getPropertyDefName().str()},
                                  {"description", prop.getSummary().str()}, {"storage_type", prop.getStorageType().str()},
                                  {"interface_type", prop.getInterfaceType().str()},
                                  {"default_value", prop.hasDefaultValue() ? llvm::json::Value(prop.getDefaultValue().str())
                                                                         : llvm::json::Value(nullptr)}});
    }
    data["properties"] = std::move(properties);
    putArray(data, "traits", std::move(traits));
    putArray(data, "regions", std::move(regions));
    putArray(data, "builders", std::move(builders));
    putArray(data, "attributes", std::move(attributes));
    putArray(data, "operands", typeConstraints(op.getOperands()));
    putArray(data, "results", typeConstraints(op.getResults()));
    if (op.hasAssemblyFormat() && !op.getAssemblyFormat().empty()) data["assembly_format"] = op.getAssemblyFormat().str();
    operations.emplace(name, std::move(data));
  }
}

void printODSContext(raw_ostream &stream) {
  Array result;
  for (auto &[name, operations] : dialects) {
    Array entries;
    for (auto &[_, operation] : operations) entries.push_back(std::move(operation));
    result.push_back(Object{{"name", name}, {"operations", std::move(entries)}});
  }
  stream << llvm::formatv("{0:2}", llvm::json::Value(Object{{"dialects", std::move(result)}}));
}

LogicalResult parseTdInclude(StringRef filename, llvm::SourceMgr &parserSrcMgr,
                             function_ref<LogicalResult(llvm::SMRange, const Twine &)> emitError) {
  // Use the source manager to open the file, but don't yet add it.
  std::string includedFile;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> includeBuffer =
      parserSrcMgr.OpenIncludeFile(filename.str(), includedFile);
  if (!includeBuffer) return emitError({}, "could not open file '" + filename + "'");

  // Setup the source manager for parsing the tablegen file.
  llvm::SourceMgr tdSrcMgr;
  tdSrcMgr.AddNewSourceBuffer(std::move(*includeBuffer), SMLoc());
  tdSrcMgr.setIncludeDirs(parserSrcMgr.getIncludeDirs());
  tdSrcMgr.setVirtualFileSystem(llvm::vfs::getRealFileSystem());

  // Set the diagnostic handler for the tablegen source manager.
  tdSrcMgr.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *rawHandlerContext) {
        const std::string *filenamePtr = static_cast<const std::string *>(rawHandlerContext);
        llvm::errs() << llvm::formatv("error while processing include file `{0}`: {1}\n",
                                      filenamePtr ? *filenamePtr : "<unknown>", diag.getMessage());
      },
      &filename);

  // Parse the tablegen file.
  llvm::RecordKeeper tdRecords;
  if (llvm::TableGenParseFile(tdSrcMgr, tdRecords)) return failure();

  // Process the parsed records.
  processTdIncludeRecords(tdRecords);
  return success();
}

} // namespace parser

int main(int argc, char **argv) {
  // FIXME: This is necessary because we link in TableGen, which defines its
  // options as static variables.. some of which overlap with our options.
  llvm::cl::ResetCommandLineParser();

  llvm::cl::list<std::string> inputFilenames(llvm::cl::Positional, llvm::cl::desc("<input files>"),
                                             llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
                                            llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs("I", llvm::cl::desc("Directory of include files"),
                                          llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::cl::opt<bool> writeIfChanged("write-if-changed", llvm::cl::desc("Only write to the output file if it changed"));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "PDLL Frontend");

  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.setVirtualFileSystem(llvm::vfs::getRealFileSystem());

  for (const auto &inputFilename : inputFilenames) {
    // Process the .td file directly
    if (failed(parser::parseTdInclude(inputFilename, sourceMgr, [](llvm::SMRange loc, const Twine &msg) {
          llvm::errs() << "error: " << msg << "\n";
          return failure();
        }))) {
      return 1;
    }
  }
  // Set up the output
  std::string outputStr;
  llvm::raw_string_ostream outputStrOS(outputStr);
  parser::printODSContext(outputStrOS);

  // Write the output.
  bool shouldWriteOutput = true;
  if (writeIfChanged) {
    // Only update the real output file if there are any differences. This
    // prevents recompilation of all the files depending on it if there aren't
    // any.
    if (auto existingOrErr = llvm::MemoryBuffer::getFile(outputFilename, /*IsText=*/true))
      if (std::move(existingOrErr.get())->getBuffer() == outputStr) shouldWriteOutput = false;
  }

  // Populate the output file if necessary.
  if (shouldWriteOutput) {
    std::string errorMessage;
    std::unique_ptr<llvm::ToolOutputFile> outputFile = openOutputFile(outputFilename, &errorMessage);
    if (!outputFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    outputFile->os() << outputStr;
    outputFile->keep();
  }

  return 0;
}
