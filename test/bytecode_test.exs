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

  for builder <- ~w(func_of_3_blocks documented_control_flow gigantic)a do
    test "bytecode round trip #{builder}", %{ctx: ctx} do
      apply(Beaver.Dummy, unquote(builder), [ctx]) |> roundtrip_bytecode()
    end
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
