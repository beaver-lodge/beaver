defmodule ASTIRTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR

  test "expander" do
    import Beaver.MLIR.AST

    defm do
      unquote(File.read!("test/ast_ir.exs") |> Code.string_to_quoted!())
    end
    |> MLIR.dump!()
  end
end
