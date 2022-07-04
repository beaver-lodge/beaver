defmodule OpPrototypeTest do
  use ExUnit.Case

  test "op compliant" do
    module = Module.concat([Beaver.MLIR.Dialect | [:TOSA, :Sub]])
    assert module == Beaver.MLIR.Dialect.TOSA.Sub
    assert Beaver.DSL.Op.Prototype.is_compliant(module)
  end
end
