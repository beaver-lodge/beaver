#ifndef ELIXIR_DIALECT
#define ELIXIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ELIXIR dialect definition.
//===----------------------------------------------------------------------===//

def Elixir_Dialect : Dialect {
    let name = "elixir";
    let summary = "A out-of-tree MLIR dialect for the Elixir programming language.";
    let description = [{
        This dialect is used to represent Elixir programs to enable high level apps like static analysis, compiler.
        Ideally, this dialect only contains ops for equivalent concepts in Elixir.Kernel.
    }];
    let cppNamespace = "::mlir::elixir";
}

//===----------------------------------------------------------------------===//
// Base elixir operation definition.
//===----------------------------------------------------------------------===//

class Elixir_Op<string mnemonic, list<Trait> traits = []> :
        Op<Elixir_Dialect, mnemonic, traits>;

#endif // ELIXIR_DIALECT
