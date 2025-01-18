defmodule ELXDialectTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen elx from ast", %{ctx: ctx} do
    ast = ElixirAST.Sample.naive_modules()

    mlir_module =
      ast
      |> ElixirAST.from_ast(ctx: ctx)
      |> Beaver.Composer.nested(
        "builtin.module",
        [
          {:nested, "func.func",
           [
             ElixirAST.MaterializeBoundVariables
           ]}
        ]
      )
      |> Beaver.Composer.run!()
      |> MLIR.verify!()

    {_, op_names} =
      mlir_module
      |> Beaver.Walker.postwalk([], fn
        %MLIR.Operation{} = op, acc -> {op, [MLIR.Operation.name(op) | acc]}
        mlir, acc -> {mlir, acc}
      end)

    op_names |> Enum.map(fn n -> assert n not in ["ex.bind", "ex.var"] end)
  end
end
