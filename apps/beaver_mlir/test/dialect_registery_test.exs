defmodule DialectRegistryTest do
  use ExUnit.Case

  alias Beaver.MLIR.Dialect

  test "example from upstream with br" do
    assert not Enum.empty?(Dialect.Registry.dialects())
    assert not Enum.empty?(Dialect.Registry.ops("arith"))
    assert Dialect.Registry.ops("cf") == ["switch", "cond_br", "br", "assert"]
  end
end
