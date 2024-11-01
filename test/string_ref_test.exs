defmodule StringRefTest do
  use Beaver.Case, async: true
  @moduletag :smoke
  alias Beaver.MLIR

  test "StringRef" do
    for _ <- 1..1_000 do
      s = "hello world"
      r = MLIR.StringRef.create(s)
      assert s == MLIR.to_string(r)
      assert s == to_string(r)
    end
  end

  test "StringRef length" do
    for size <- 1..1000 do
      :rand.seed(:exsss, {100, 101, 102})
      s = 0..size |> Enum.shuffle() |> List.to_string()
      r = MLIR.StringRef.create(s)
      assert s == MLIR.to_string(r)
      assert byte_size(s) == MLIR.StringRef.length(r)
    end
  end

  describe "printer" do
    test "collect string ref", %{ctx: ctx} do
      {:ok, txt} =
        Beaver.Printer.run(fn cb, ud ->
          MLIR.Location.file(name: "1", line: 2, column: 3, ctx: ctx)
          |> MLIR.CAPI.mlirLocationPrint(cb, ud)
        end)

      assert ~s{loc("1":2:3)} == txt

      {attr, "1 : i64"} =
        Beaver.Printer.run(fn cb, ud ->
          MLIR.Attribute.get("1", ctx: ctx)
          |> tap(&MLIR.CAPI.mlirAttributePrint(&1, cb, ud))
        end)

      refute MLIR.null?(attr)
    end

    test "flush only once", %{ctx: ctx} do
      {sp, user_data} = Beaver.Printer.create()

      MLIR.Location.file(name: "1", line: 2, column: 3, ctx: ctx)
      |> MLIR.CAPI.mlirLocationPrint(Beaver.Printer.callback(), user_data)

      assert ~s{loc("1":2:3)} == Beaver.Printer.flush(sp)
      assert_raise Kinda.CallError, ~r"Already flushed", fn -> Beaver.Printer.flush(sp) end
    end
  end
end
