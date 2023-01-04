defmodule PrototypeRegistryTest do
  use ExUnit.Case

  test "test insert" do
    Beaver.DSL.Op.Registry.register("test", Test.Test)
    assert Beaver.DSL.Op.Registry.lookup("test") == Test.Test
  end
end
