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
end
