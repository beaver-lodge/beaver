defmodule ELXDialectTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen elx from ast", _test_context do
    String
    |> BeamFile.elixir_quoted!()
    # |> tap(&IO.puts/1)
    # |> tap(&File.write("ast", inspect(&1, limit: :infinity, pretty: true)))
    |> ElixirAST.from_ast()
  end
end
