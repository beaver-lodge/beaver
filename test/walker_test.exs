defmodule WalkerTest do
  use Beaver.Case, async: true

  test "access attributes", test_context do
    ctx = test_context[:ctx]
    m = Beaver.Dummy.func_of_3_blocks(ctx)
    f = m |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)

    by_atom =
      Beaver.Walker.attributes(f)[:sym_name]
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> to_string()

    by_str =
      Beaver.Walker.attributes(f)["sym_name"]
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> to_string()

    assert by_atom == by_str
  end
end
