defmodule ELXDialectTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen elx from ast", _test_context do
    quote do
      defmodule ElixirASTBasics do
        def add_two_literal() do
          a = 1 + 2
          b = a + 3
          b
        end
      end
    end
    |> ElixirAST.from_ast()
  end
end
