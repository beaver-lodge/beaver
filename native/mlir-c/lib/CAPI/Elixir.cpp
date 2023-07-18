#include "mlir-c/Dialect/Elixir.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Elixir/IR/ElixirDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Elixir, elixir,
                                      mlir::elixir::ElixirDialect)
