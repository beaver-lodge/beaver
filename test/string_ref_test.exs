defmodule StringRefTest do
  use Beaver.Case, async: true
  @moduletag :smoke
  alias Beaver.MLIR

  test "StringRef" do
    for _ <- 1..1_000 do
      s = "hello world"
      r = MLIR.StringRef.create(s)
      assert s == MLIR.StringRef.to_string(r)
      assert s == to_string(r)
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

  describe "printer" do
    test "collect string ref", %{ctx: ctx} do
      {:ok, txt} =
        Beaver.StringPrinter.run(fn cb, ud ->
          MLIR.Location.file(name: "1", line: 2, column: 3, ctx: ctx)
          |> MLIR.CAPI.mlirLocationPrint(cb, ud)
        end)

      assert ~s{loc("1":2:3)} == txt

      {attr, "1 : i64"} =
        Beaver.StringPrinter.run(fn cb, ud ->
          MLIR.Attribute.get("1", ctx: ctx)
          |> tap(&MLIR.CAPI.mlirAttributePrint(&1, cb, ud))
        end)

      refute MLIR.is_null(attr)
    end

    test "flush only once", %{ctx: ctx} do
      {sp, user_data} = Beaver.StringPrinter.create()

      MLIR.Location.file(name: "1", line: 2, column: 3, ctx: ctx)
      |> MLIR.CAPI.mlirLocationPrint(Beaver.StringPrinter.callback(), user_data)

      assert ~s{loc("1":2:3)} == Beaver.StringPrinter.flush(sp)
      assert_raise Kinda.CallError, ~r"Already flushed", fn -> Beaver.StringPrinter.flush(sp) end
    end
  end
end
