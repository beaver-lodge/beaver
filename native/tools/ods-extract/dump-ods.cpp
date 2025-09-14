#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Tools/PDLL/ODS/Constraint.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/ODS/Dialect.h"
#include "mlir/Tools/PDLL/ODS/Operation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Parser.h"

using namespace mlir;
using namespace mlir::pdll;

namespace parser {

template <typename T>
SmallVector<T *> sortMapByName(const llvm::StringMap<std::unique_ptr<T>> &map) {
  SmallVector<T *> storage;
  for (auto &entry : map)
    storage.push_back(entry.second.get());
  llvm::sort(storage, [](const auto &lhs, const auto &rhs) {
    return lhs->getName() < rhs->getName();
  });
  return storage;
}

void printODSContext(raw_ostream &os, const ods::Context &odsContext) {
  using namespace mlir::pdll::ods;
  auto getVariableLengthStr = [](VariableLengthKind kind) -> StringRef {
    switch (kind) {
    case VariableLengthKind::Optional:
      return "Optional";
    case VariableLengthKind::Single:
      return "Single";
    case VariableLengthKind::Variadic:
      return "Variadic";
    }
    llvm_unreachable("unknown variable length kind");
  };

  llvm::json::OStream j(os, 2);

  auto printOperandsOrResults = [&](StringRef name,
                                    ArrayRef<OperandOrResult> elements) {
    if (elements.empty())
      return;
    j.attributeArray(name, [&] {
      for (const auto &element : elements) {
        j.object([&] {
          j.attribute("name", element.getName());
          j.attribute("constraint", element.getConstraint().getDemangledName());
          j.attribute("description", element.getConstraint().getSummary());
          j.attribute("kind",
                      getVariableLengthStr(element.getVariableLengthKind()));
        });
      }
    });
  };

  j.object([&] {
    j.attributeArray("dialects", [&] {
      for (const Dialect &dialect : odsContext.getDialects()) {
        j.object([&] {
          j.attribute("name", dialect.getName());
          j.attributeArray("operations", [&] {
            for (const Operation *op : sortMapByName(dialect.getOperations())) {
              j.object([&] {
                j.attribute("name", op->getName());
                j.attribute("summary", op->getSummary());
                j.attribute("description", op->getDescription());
                j.attribute("result_type_inference",
                            op->hasResultTypeInferrence());

                // Attributes
                ArrayRef<Attribute> attributes = op->getAttributes();
                if (!attributes.empty()) {
                  j.attributeArray("attributes", [&] {
                    for (const Attribute &attr : attributes) {
                      j.object([&] {
                        j.attribute("name", attr.getName());
                        j.attribute("constraint",
                                    attr.getConstraint().getDemangledName());
                        j.attribute("description",
                                    attr.getConstraint().getSummary());
                        j.attribute("kind",
                                    getVariableLengthStr(
                                        attr.isOptional()
                                            ? VariableLengthKind::Optional
                                            : VariableLengthKind::Single));
                      });
                    }
                  });
                }

                // Operands
                printOperandsOrResults("operands", op->getOperands());

                // Results
                printOperandsOrResults("results", op->getResults());
              });
            }
          });
        });
      }
    });
  });
}

std::string processAndFormatDoc(const Twine &doc) {

  std::string docStr;
  {
    llvm::raw_string_ostream docOS(docStr);
    std::string tmpDocStr = doc.str();
    raw_indented_ostream(docOS).printReindented(
        StringRef(tmpDocStr).rtrim(" 	"));
  }
  return docStr;
}
void processTdIncludeRecords(const llvm::RecordKeeper &tdRecords,
                             ods::Context &odsContext) {
  // Return the length kind of the given value.
  auto getLengthKind = [](const auto &value) {
    if (value.isOptional())
      return ods::VariableLengthKind::Optional;
    return value.isVariadic() ? ods::VariableLengthKind::Variadic
                              : ods::VariableLengthKind::Single;
  };

  auto addTypeConstraint = [&](const tblgen::NamedTypeConstraint &cst)
      -> const ods::TypeConstraint & {
    return odsContext.insertTypeConstraint(cst.constraint.getUniqueDefName(),
                                           (cst.constraint.getSummary()),
                                           cst.constraint.getCppType());
  };

  // Process the parsed tablegen records to build ODS information.
  /// Operations.
  for (const llvm::Record *def : tdRecords.getAllDerivedDefinitions("Op")) {
    tblgen::Operator op(def);

    // Check to see if this operation is known to support type inferrence.
    bool supportsResultTypeInferrence =
        op.getTrait("::mlir::InferTypeOpInterface::Trait");

    auto [odsOp, inserted] = odsContext.insertOperation(
        op.getOperationName(), op.getSummary(),
        processAndFormatDoc(op.getDescription()), op.getQualCppClassName(),
        supportsResultTypeInferrence, op.getLoc().front());

    // Ignore operations that have already been added.
    if (!inserted)
      continue;

    for (const tblgen::NamedAttribute &attr : op.getAttributes()) {
      odsOp->appendAttribute(
          attr.name, attr.attr.isOptional(),
          odsContext.insertAttributeConstraint(attr.attr.getUniqueDefName(),
                                               attr.attr.getSummary(),
                                               attr.attr.getStorageType()));
    }
    for (const tblgen::NamedTypeConstraint &operand : op.getOperands()) {
      odsOp->appendOperand(operand.name, getLengthKind(operand),
                           addTypeConstraint(operand));
    }
    for (const tblgen::NamedTypeConstraint &result : op.getResults()) {
      odsOp->appendResult(result.name, getLengthKind(result),
                          addTypeConstraint(result));
    }
  }
}

LogicalResult parseTdInclude(
    ods::Context &odsContext, StringRef filename, llvm::SourceMgr &parserSrcMgr,
    function_ref<LogicalResult(llvm::SMRange, const Twine &)> emitError) {
  // Use the source manager to open the file, but don't yet add it.
  std::string includedFile;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> includeBuffer =
      parserSrcMgr.OpenIncludeFile(filename.str(), includedFile);
  if (!includeBuffer)
    return emitError({}, "could not open file '" + filename + "'");

  // Setup the source manager for parsing the tablegen file.
  llvm::SourceMgr tdSrcMgr;
  tdSrcMgr.AddNewSourceBuffer(std::move(*includeBuffer), SMLoc());
  tdSrcMgr.setIncludeDirs(parserSrcMgr.getIncludeDirs());

  // Set the diagnostic handler for the tablegen source manager.
  tdSrcMgr.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *rawHandlerContext) {
        const std::string *filenamePtr =
            static_cast<const std::string *>(rawHandlerContext);
        llvm::errs() << llvm::formatv(
            "error while processing include file `{0}`: {1}\n",
            filenamePtr ? *filenamePtr : "<unknown>", diag.getMessage());
      },
      &filename);

  // Parse the tablegen file.
  llvm::RecordKeeper tdRecords;
  if (llvm::TableGenParseFile(tdSrcMgr, tdRecords))
    return failure();

  // Process the parsed records.
  processTdIncludeRecords(tdRecords, odsContext);
  return success();
}

} // namespace parser

int main(int argc, char **argv) {
  // FIXME: This is necessary because we link in TableGen, which defines its
  // options as static variables.. some of which overlap with our options.
  llvm::cl::ResetCommandLineParser();

  llvm::cl::list<std::string> inputFilenames(llvm::cl::Positional,
                                             llvm::cl::desc("<input files>"),
                                             llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::cl::opt<bool> writeIfChanged(
      "write-if-changed",
      llvm::cl::desc("Only write to the output file if it changed"));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "PDLL Frontend");

  ods::Context odsContext;
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);

  for (const auto &inputFilename : inputFilenames) {
    // Process the .td file directly
    if (failed(parser::parseTdInclude(odsContext, inputFilename, sourceMgr,
                                      [](llvm::SMRange loc, const Twine &msg) {
                                        llvm::errs()
                                            << "error: " << msg << "\n";
                                        return failure();
                                      }))) {
      return 1;
    }
  }
  // Set up the output
  std::string outputStr;
  llvm::raw_string_ostream outputStrOS(outputStr);
  parser::printODSContext(outputStrOS, odsContext);

  // Write the output.
  bool shouldWriteOutput = true;
  if (writeIfChanged) {
    // Only update the real output file if there are any differences. This
    // prevents recompilation of all the files depending on it if there aren't
    // any.
    if (auto existingOrErr =
            llvm::MemoryBuffer::getFile(outputFilename, /*IsText=*/true))
      if (std::move(existingOrErr.get())->getBuffer() == outputStr)
        shouldWriteOutput = false;
  }

  // Populate the output file if necessary.
  if (shouldWriteOutput) {
    std::string errorMessage;
    std::unique_ptr<llvm::ToolOutputFile> outputFile =
        openOutputFile(outputFilename, &errorMessage);
    if (!outputFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    outputFile->os() << outputStr;
    outputFile->keep();
  }

  return 0;
}
