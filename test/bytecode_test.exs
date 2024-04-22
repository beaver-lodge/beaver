defmodule BytecodeTest do
  use Beaver.Case, async: true

  @moduletag :smoke

  defp roundtrip_bytecode(m) do
    m
    |> MLIR.Operation.verify!()
    |> MLIR.to_string(bytecode: true)
    |> tap(fn s -> assert String.starts_with?(s, "ML\xefR") end)
    |> then(&MLIR.Module.create!(MLIR.CAPI.mlirModuleGetContext(m), &1))
    |> MLIR.Operation.verify!()
  end

  test "bytecode writing and parsing", test_context do
    Beaver.Dummy.func_of_3_blocks(test_context[:ctx])
    |> roundtrip_bytecode
  end

  test "bytecode writing and parsing readme", test_context do
    Beaver.Dummy.readme(test_context[:ctx])
    |> roundtrip_bytecode
  end

  test "bytecode writing and parsing gigantic", test_context do
    Beaver.Dummy.gigantic(test_context[:ctx])
    |> roundtrip_bytecode
  end
end
