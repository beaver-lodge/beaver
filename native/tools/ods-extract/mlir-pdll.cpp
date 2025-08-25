//===- mlir-pdll.cpp - MLIR PDLL frontend -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include <set>
#include <string>

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

/// The desired output type.
enum class OutputType {
  AST,
  MLIR,
  CPP,
};

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
                ArrayRef<OperandOrResult> operands = op->getOperands();
                if (!operands.empty()) {
                  j.attributeArray("operands", [&] {
                    for (const OperandOrResult &operand : operands) {
                      j.object([&] {
                        j.attribute("name", operand.getName());
                        j.attribute("constraint",
                                    operand.getConstraint().getDemangledName());
                        j.attribute("description",
                                    operand.getConstraint().getSummary());
                        j.attribute("kind",
                                    getVariableLengthStr(
                                        operand.getVariableLengthKind()));
                      });
                    }
                  });
                }

                // Results
                ArrayRef<OperandOrResult> results = op->getResults();
                if (!results.empty()) {
                  j.attributeArray("results", [&] {
                    for (const OperandOrResult &result : results) {
                      j.object([&] {
                        j.attribute("name", result.getName());
                        j.attribute("constraint",
                                    result.getConstraint().getDemangledName());
                        j.attribute("description",
                                    result.getConstraint().getSummary());
                        j.attribute("kind",
                                    getVariableLengthStr(
                                        result.getVariableLengthKind()));
                      });
                    }
                  });
                }
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
    raw_indented_ostream(docOS).printReindented(
        doc.getSingleStringRef().rtrim(" \t"));
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

  auto shouldBeSkipped = [](const llvm::Record *def) {
    return def->isAnonymous() || def->isSubClassOf("DeclareInterfaceMethods");
  };

  /// Attr constraints.
  for (const llvm::Record *def : tdRecords.getAllDerivedDefinitions("Attr")) {
    if (shouldBeSkipped(def))
      continue;

    tblgen::Attribute constraint(def);
  }
  /// Type constraints.
  for (const llvm::Record *def : tdRecords.getAllDerivedDefinitions("Type")) {
    if (shouldBeSkipped(def))
      continue;

    tblgen::TypeConstraint constraint(def);
  }
  /// OpInterfaces.
  for (const llvm::Record *def :
       tdRecords.getAllDerivedDefinitions("OpInterface")) {
    if (shouldBeSkipped(def))
      continue;

    std::string cppClassName =
        llvm::formatv("{0}::{1}", def->getValueAsString("cppNamespace"),
                      def->getValueAsString("cppInterfaceName"))
            .str();
    std::string codeBlock =
        llvm::formatv("return ::mlir::success(llvm::isa<{0}>(self));",
                      cppClassName)
            .str();

    std::string desc =
        processAndFormatDoc(def->getValueAsString("description"));
  }
}

LogicalResult parseTdInclude(
    ods::Context &odsContext, StringRef filename, llvm::SourceMgr &parserSrcMgr,
    function_ref<LogicalResult(llvm::SMRange, const Twine &)> emitError) {
  // Use the source manager to open the file, but don't yet add it.
  std::string includedFile;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> includeBuffer =
      parserSrcMgr.OpenIncludeFile(filename.str(), includedFile);

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

static LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
              OutputType outputType, std::vector<std::string> &includeDirs,
              bool dumpODS, std::set<std::string> *includedFiles) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(chunkBuffer), SMLoc());

  ods::Context odsContext;

  // Process each buffer in the source manager (skipping buffer 0 which is the
  // main file)
  for (unsigned i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(i + 1);
    std::string filename = buffer->getBufferIdentifier().str();

    // Read each line from the buffer
    StringRef bufferContent = buffer->getBuffer();
    SmallVector<StringRef> lines;
    bufferContent.split(lines, '\n');

    for (StringRef line : lines) {
      line = line.trim();
      // Skip empty lines and comments
      if (line.empty() || line.starts_with("//"))
        continue;

      // Process .td file references
      if (line.ends_with(".td")) {
        if (failed(parser::parseTdInclude(
                odsContext, line, sourceMgr,
                [](llvm::SMRange loc, const Twine &msg) {
                  llvm::errs() << "error: " << msg << "\n";
                  return failure();
                }))) {
          return failure();
        }
      }
    }
  }
  parser::printODSContext(os, odsContext);
  return success();
}

int main(int argc, char **argv) {
  // FIXME: This is necessary because we link in TableGen, which defines its
  // options as static variables.. some of which overlap with our options.
  llvm::cl::ResetCommandLineParser();

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::cl::opt<bool> dumpODS(
      "dump-ods",
      llvm::cl::desc(
          "Print out the parsed ODS information from the input file"),
      llvm::cl::init(false));
  llvm::cl::opt<std::string> inputSplitMarker{
      "split-input-file", llvm::cl::ValueOptional,
      llvm::cl::callback([&](const std::string &str) {
        // Implicit value: use default marker if flag was used without value.
        if (str.empty())
          inputSplitMarker.setValue(kDefaultSplitMarker);
      }),
      llvm::cl::desc("Split the input file into chunks using the given or "
                     "default marker and process each chunk independently"),
      llvm::cl::init("")};
  llvm::cl::opt<std::string> outputSplitMarker(
      "output-split-marker",
      llvm::cl::desc("Split marker to use for merging the ouput"),
      llvm::cl::init(kDefaultSplitMarker));
  llvm::cl::opt<enum OutputType> outputType(
      "x", llvm::cl::init(OutputType::AST),
      llvm::cl::desc("The type of output desired"),
      llvm::cl::values(clEnumValN(OutputType::AST, "ast",
                                  "generate the AST for the input file"),
                       clEnumValN(OutputType::MLIR, "mlir",
                                  "generate the PDL MLIR for the input file"),
                       clEnumValN(OutputType::CPP, "cpp",
                                  "generate a C++ source file containing the "
                                  "patterns for the input file")));
  llvm::cl::opt<std::string> dependencyFilename(
      "d", llvm::cl::desc("Dependency filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init(""));
  llvm::cl::opt<bool> writeIfChanged(
      "write-if-changed",
      llvm::cl::desc("Only write to the output file if it changed"));

  // `ResetCommandLineParser` at the above unregistered the "D" option
  // of `llvm-tblgen`, which causes tblgen usage to fail due to
  // "Unknnown command line argument '-D...`" when a macros name is
  // present. The following is a workaround to re-register it again.
  llvm::cl::list<std::string> macroNames(
      "D",
      llvm::cl::desc("Name of the macro to be defined -- ignored by mlir-pdll"),
      llvm::cl::value_desc("macro name"), llvm::cl::Prefix);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "PDLL Frontend");

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // If we are creating a dependency file, we'll also need to track what files
  // get included during processing.
  std::set<std::string> includedFilesStorage;
  std::set<std::string> *includedFiles = nullptr;
  if (!dependencyFilename.empty())
    includedFiles = &includedFilesStorage;

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  std::string outputStr;
  llvm::raw_string_ostream outputStrOS(outputStr);
  auto processFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                       raw_ostream &os) {
    // Split does not guarantee null-termination. Make a copy of the buffer to
    // ensure null-termination.
    if (!chunkBuffer->getBuffer().ends_with('\0')) {
      chunkBuffer = llvm::MemoryBuffer::getMemBufferCopy(
          chunkBuffer->getBuffer(), chunkBuffer->getBufferIdentifier());
    }
    return processBuffer(os, std::move(chunkBuffer), outputType, includeDirs,
                         dumpODS, includedFiles);
  };
  if (failed(splitAndProcessBuffer(std::move(inputFile), processFn, outputStrOS,
                                   inputSplitMarker, outputSplitMarker)))
    return 1;

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
