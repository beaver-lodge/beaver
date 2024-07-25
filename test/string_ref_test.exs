defmodule StringRefTest do
  use ExUnit.Case, async: true
  @moduletag :smoke
  alias Beaver.MLIR

  test "StringRef" do
    for _ <- 1..1_000 do
      s = "hello world"
      r = MLIR.StringRef.create(s)
      assert s == MLIR.StringRef.to_string(r)
    end
  end

  test "StringRef length" do
    for size <- 1..1000 do
      :rand.seed(:exsss, {100, 101, 102})
      s = 0..size |> Enum.shuffle() |> List.to_string()
      r = MLIR.StringRef.create(s)
      assert s == MLIR.StringRef.to_string(r)
      assert byte_size(s) == MLIR.StringRef.length(r)
    end
  end
end
