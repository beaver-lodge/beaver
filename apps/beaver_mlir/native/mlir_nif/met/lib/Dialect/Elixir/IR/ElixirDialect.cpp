#include "mlir/Dialect/Elixir/IR/ElixirDialect.h"
#include "mlir/Dialect/Elixir/IR/ElixirOps.h"

using namespace mlir;
using namespace mlir::elixir;

#include "mlir/Dialect/Elixir/IR/ElixirOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Elixir dialect.
//===----------------------------------------------------------------------===//

void ElixirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Elixir/IR/ElixirOps.cpp.inc"
      >();
}
