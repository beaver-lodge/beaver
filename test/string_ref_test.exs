defmodule StringRefTest do
  use ExUnit.Case, async: true
  @moduletag :smoke
  alias Beaver.MLIR

  test "StringRef" do
    require Logger

    for _ <- 1..1_000 do
      s = "hello world"
      r = MLIR.StringRef.create(s)
      assert s == MLIR.StringRef.to_string(r)
    end
  end
end
