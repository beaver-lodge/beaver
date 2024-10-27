defmodule DiagnosticTest do
  use Beaver.Case, async: true, diagnostic: :server
  alias Beaver.MLIR.Attribute

  describe "server" do
    test "handler", %{ctx: ctx} do
      {:ok, server} =
        GenServer.start(
          Beaver.DiagnosticHandler,
          &"#{&2}--#{to_string(MLIR.location(&1))}--#{to_string(&1)}"
        )

      handler_id = Beaver.Diagnostic.attach(ctx, server)

      assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
        Attribute.get("???", ctx: ctx) |> MLIR.is_null()
      end

      assert Beaver.DiagnosticHandler.collect(server) ==
               "--???:1:1--expected attribute value"

      :ok = GenServer.stop(server)
      Beaver.Diagnostic.detach(ctx, handler_id)
    end

    test "handler with init state", %{ctx: ctx} do
      {:ok, server} =
        GenServer.start(
          Beaver.DiagnosticHandler,
          {fn -> "hello" end,
           &"#{&2}--#{MLIR.Diagnostic.severity(&1)}--#{to_string(MLIR.location(&1))}--#{to_string(&1)}"}
        )

      handler_id = Beaver.Diagnostic.attach(ctx, server)

      assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
        Attribute.get("???", ctx: ctx) |> MLIR.is_null()
      end

      assert Beaver.DiagnosticHandler.collect(server) ==
               "hello--error--???:1:1--expected attribute value"

      :ok = GenServer.stop(server)
      Beaver.Diagnostic.detach(ctx, handler_id)
    end
  end
end
