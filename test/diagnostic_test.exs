defmodule DiagnosticTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR

  test "handler", %{ctx: ctx} do
    assert_raise ArgumentError,
                 "fail to parse attribute\ninvalid_attr:1:1: expected attribute value",
                 fn ->
                   MLIR.Attribute.get("invalid_attr", ctx: ctx)
                 end
  end
end
