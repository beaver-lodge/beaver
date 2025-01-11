defmodule DiagnosticTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR

  test "handler", %{ctx: ctx} do
    assert_raise ArgumentError,
                 "fail to parse attribute\nat invalid_attr:1:1: expected attribute value",
                 fn ->
                   MLIR.Attribute.get("invalid_attr", ctx: ctx)
                 end
  end

  @tag :stderr
  test "emit", %{ctx: ctx} do
    loc = MLIR.Location.unknown(ctx: ctx)
    MLIR.Diagnostic.emit(loc, "some error msg")
  end
end
