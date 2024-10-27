defmodule ELXDialectTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen elx from ast", %{ctx: ctx} do
    ast =
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

    mlir_module =
      ast
      |> ElixirAST.from_ast(ctx: ctx)
      |> MLIR.Pass.Composer.nested(
        "builtin.module",
        [
          {:nested, "func.func",
           [
             ElixirAST.MaterializeBoundVariables
           ]}
        ]
      )
      |> MLIR.Pass.Composer.run!()
      |> MLIR.Operation.verify!()

    {_, op_names} =
      mlir_module
      |> Beaver.Walker.postwalk([], fn
        %MLIR.Operation{} = op, acc -> {op, [MLIR.Operation.name(op) | acc]}
        mlir, acc -> {mlir, acc}
      end)

    op_names |> Enum.map(fn n -> assert n not in ["ex.bind", "ex.var"] end)
  end
end
