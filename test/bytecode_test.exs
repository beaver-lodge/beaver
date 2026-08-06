defmodule BytecodeTest do
  use Beaver.Case, async: true

  @moduletag :smoke

  defp roundtrip_bytecode(m) do
    m
    |> MLIR.verify!()
    |> MLIR.to_string(bytecode: true)
    |> tap(fn s -> assert String.starts_with?(s, "ML\xefR") end)
    |> then(&MLIR.Module.create!(&1, ctx: MLIR.context(m)))
    |> MLIR.verify!()
  end

  test "bytecode writing and parsing", %{ctx: ctx} do
    Beaver.Dummy.func_of_3_blocks(ctx)
    |> roundtrip_bytecode
  end

  test "bytecode writing and parsing readme", %{ctx: ctx} do
    Beaver.Dummy.readme(ctx)
    |> roundtrip_bytecode
  end

  test "bytecode writing and parsing gigantic", %{ctx: ctx} do
    Beaver.Dummy.gigantic(ctx)
    |> roundtrip_bytecode
  end

  test "bytecode writer accepts an explicit desired emit version", %{ctx: ctx} do
    module = MLIR.Module.create!("module {}", ctx: ctx)

    assert {:ok, bytecode} = MLIR.Bytecode.write(module, desired_emit_version: 0)
    assert String.starts_with?(bytecode, "ML\xEFR")

    roundtripped = MLIR.Bytecode.read!(bytecode, ctx: ctx)
    assert MLIR.verify?(roundtripped)

    MLIR.Module.destroy(roundtripped)
    MLIR.Module.destroy(module)
  end
end
