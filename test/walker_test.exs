defmodule WalkerTest do
  use Beaver.Case, async: true

  test "access", %{ctx: ctx} do
    m = Beaver.Dummy.func_of_3_blocks(ctx)
    f = m |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)
    sym_name = MLIR.SymbolTable.attribute_name()

    assert Beaver.Walker.attributes(f)[sym_name] |> to_string() ==
             Beaver.Walker.attributes(f)[to_string(sym_name)] |> to_string()

    assert nil == Beaver.Walker.attributes(f)[:foo]
    r0 = Beaver.Walker.regions(f)[0]
    b0 = Beaver.Walker.blocks(r0)[0]
    assert "arith.constant" = Beaver.Walker.operations(b0)[0] |> MLIR.Operation.name()
    b1 = Beaver.Walker.blocks(r0)[1]
    a = Beaver.Walker.arguments(b1)[0]
    assert %MLIR.OpOperand{} = Beaver.Walker.uses(a)[0]
  end
end
