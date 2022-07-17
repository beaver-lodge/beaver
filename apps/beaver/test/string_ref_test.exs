defmodule StringRefTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR

  test "StringRef" do
    s = "hello world"
    r = MLIR.StringRef.create(s) |> IO.inspect()
    assert s == MLIR.StringRef.extract(r)
  end
end
