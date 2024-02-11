defmodule ASTIRTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR

  test "expander", test_context do
    defmodule ASTExample do
      import Beaver.MLIR.AST
      @file "test/ast_ir.exs"

      defm gen_ir() do
        unquote(File.read!("test/ast_ir.exs") |> Code.string_to_quoted!())
      end
    end

    ctx = test_context[:ctx]
    MLIR.CAPI.mlirContextSetAllowUnregisteredDialects(ctx, true)
    m = ASTExample.gen_ir(ctx) |> MLIR.dump!()
    m |> MLIR.Operation.verify!()
  end
end
