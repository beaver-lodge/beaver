defmodule WalkerTest do
  use Beaver.Case, async: true

  test "indexed slicing and suspended linked iteration", %{ctx: ctx} do
    module =
      MLIR.Module.create!("module { func.func @f(%a: i32, %b: i32, %c: i32) { return } }",
        ctx: ctx
      )

    function = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)

    block =
      function |> Beaver.Walker.regions() |> Enum.at(0) |> Beaver.Walker.blocks() |> Enum.at(0)

    arguments = Beaver.Walker.arguments(block)
    [a, _b, c] = Enum.to_list(arguments)
    assert [first, last] = Enum.slice(arguments, 0..2//2)
    assert MLIR.equal?(first, a)
    assert MLIR.equal?(last, c)
    assert Enum.empty?(Beaver.Walker.results(function))
    assert Enum.member?(arguments, c)

    operations = Beaver.Walker.operations(block)
    reducer = fn op, acc -> {:suspend, [MLIR.Operation.name(op) | acc]} end

    for walker <- [arguments, operations] do
      unreadable = %{
        walker
        | get_num: fn _ -> flunk("read count before iteration resumed") end,
          get_first: fn _ -> flunk("read first element before iteration resumed") end
      }

      assert {:halted, :stopped} = Enumerable.reduce(unreadable, {:halt, :stopped}, reducer)
      assert {:suspended, [], resume} = Enumerable.reduce(unreadable, {:suspend, []}, reducer)
      assert {:halted, :stopped} = resume.({:halt, :stopped})
    end

    assert {:suspended, [], resume} = Enumerable.reduce(operations, {:suspend, []}, reducer)
    assert {:suspended, ["func.return"], resume} = resume.({:cont, []})
    assert {:done, ["func.return"]} = resume.({:cont, ["func.return"]})
    assert {:halted, :stopped} = Enumerable.reduce(operations, {:halt, :stopped}, reducer)
    MLIR.Module.destroy(module)
  end

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
