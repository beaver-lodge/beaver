defmodule SymbolTableTest do
  use Beaver.Case, async: true
  @moduletag :smoke
  alias Beaver.MLIR

  test "accesses semantic symbol visibility", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        "module { func.func private @example() { return } }",
        ctx: ctx
      )

    symbol = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)

    assert MLIR.SymbolTable.default_visibility_attribute_name() == :sym_visibility
    assert MLIR.SymbolTable.visibility(symbol) == :private

    assert :ok = MLIR.SymbolTable.set_visibility(symbol, :nested)
    assert MLIR.SymbolTable.visibility(symbol) == :nested

    assert :ok = MLIR.SymbolTable.set_visibility(symbol, :public)
    assert MLIR.SymbolTable.visibility(symbol) == :public
    refute MLIR.to_string(symbol) =~ "sym_visibility"
  end

  test "rejects unknown visibility", %{ctx: ctx} do
    module = MLIR.Module.create!("module { func.func @example() { return } }", ctx: ctx)
    symbol = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)

    assert_raise ArgumentError, "invalid symbol visibility: :package", fn ->
      apply(MLIR.SymbolTable, :set_visibility, [symbol, :package])
    end
  end
end
