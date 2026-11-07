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

    for walker <- [
          Beaver.Walker.blocks(r0),
          Beaver.Walker.operations(b0),
          Beaver.Walker.arguments(b1)
        ] do
      items = Enum.to_list(walker)
      assert Enum.zip(walker, items) |> Enum.all?(fn {a, b} -> MLIR.equal?(a, b) end)
      assert length(Enum.slice(walker, 0, 2)) == min(2, length(items))
      assert Enum.reduce_while(walker, 0, fn _, n -> {:halt, n + 1} end) == 1

      reducer = fn item, acc -> {:suspend, [item | acc]} end
      assert {:suspended, [], resume} = Enumerable.reduce(walker, {:suspend, []}, reducer)
      assert {:suspended, [first], resume} = resume.({:cont, []})
      assert MLIR.equal?(first, hd(items))
      assert {:halted, :stopped} = resume.({:halt, :stopped})
    end
  end
end
