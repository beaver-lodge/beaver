defmodule ELXDialectTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen elx from ast", test_context do
    quote do
      defmodule TwoFuncMod do
        def add_two_literal() do
          a = 1 + 2
          b = a + 3
          b
        end

        def main() do
          add_two_literal()
        end
      end

      defmodule OneFuncMod do
        def add_two_literal() do
          a = 1 + 2
          b = a + 3
          b
        end
      end
    end
    |> ElixirAST.from_ast(ctx: test_context[:ctx])
    |> MLIR.dump!()
    |> MLIR.Pass.Composer.nested(
      "builtin.module",
      [
        {:nested, "func.func",
         [
           ElixirAST.UseBoundVariables
         ]}
      ]
    )
    |> MLIR.Pass.Composer.run!()
    |> MLIR.Operation.verify!()
    |> MLIR.dump!()
  end
end
