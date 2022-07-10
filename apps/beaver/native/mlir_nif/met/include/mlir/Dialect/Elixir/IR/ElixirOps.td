#ifndef ELIXIR_OPS
#define ELIXIR_OPS

include "mlir/Dialect/Elixir/IR/ElixirDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Elixir_AddOp : Elixir_Op<"add", [NoSideEffect]> {
    let summary = "+/2 in Elixir.Kernel";
    let arguments = (ins AnyType:$left, AnyType:$right);
    let results = (outs AnyType:$res);
}

#endif // ELIXIR_OPS
