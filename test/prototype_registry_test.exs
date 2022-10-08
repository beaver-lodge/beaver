defmodule PrototypeRegistryTest do
  use ExUnit.Case

  test "test insert" do
    Beaver.MLIR.DSL.Op.Registry.register("test", Test.Test)
    assert Beaver.MLIR.DSL.Op.Registry.lookup("test") == Test.Test
  end
end
