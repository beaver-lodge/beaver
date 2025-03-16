#ifndef ELIXIR_ELIXIROPS_H
#define ELIXIR_ELIXIROPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Elixir/IR/ElixirOps.h.inc"

#endif // ELIXIR_ELIXIROPS_H
