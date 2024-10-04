defmodule WalkerTest do
  use Beaver.Case, async: true

  test "access attributes", test_context do
    ctx = test_context[:ctx]
    m = Beaver.Dummy.func_of_3_blocks(ctx)
    f = m |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)

    assert Beaver.Walker.attributes(f)[:sym_name] |> to_string() ==
             Beaver.Walker.attributes(f)["sym_name"] |> to_string()
  end
end
