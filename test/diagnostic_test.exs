defmodule DiagnosticTest do
  use Beaver.Case, async: true, diagnostic: :server
  alias Beaver.MLIR.Attribute

  describe "server" do
    test "handler", %{ctx: ctx} do
      {:ok, server} =
        GenServer.start(
          Beaver.DiagnosticHandlerRunner,
          &"#{&2}--#{to_string(MLIR.location(&1))}--#{to_string(&1)}"
        )

      handler_id = Beaver.DiagnosticHandlerRunner.attach(ctx, server)

      assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
        Attribute.get("???", ctx: ctx) |> MLIR.is_null()
      end

      assert Beaver.DiagnosticHandlerRunner.collect(server) ==
               "--???:1:1--expected attribute value"

      :ok = GenServer.stop(server)
      Beaver.MLIR.Diagnostic.detach(ctx, handler_id)
    end

    test "handler with init state", %{ctx: ctx} do
      {:ok, server} =
        GenServer.start(
          Beaver.DiagnosticHandlerRunner,
          {fn -> "hello" end,
           &"#{&2}--#{MLIR.Diagnostic.severity(&1)}--#{to_string(MLIR.location(&1))}--#{to_string(&1)}"}
        )

      handler_id = Beaver.DiagnosticHandlerRunner.attach(ctx, server)

      assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
        Attribute.get("???", ctx: ctx) |> MLIR.is_null()
      end

      assert Beaver.DiagnosticHandlerRunner.collect(server) ==
               "hello--error--???:1:1--expected attribute value"

      :ok = GenServer.stop(server)
      Beaver.MLIR.Diagnostic.detach(ctx, handler_id)
    end

    test "with_diagnostics", %{ctx: ctx} do
      {%RuntimeError{}, txt} =
        Beaver.with_diagnostics(
          ctx,
          fn ->
            assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
              Attribute.get("???", ctx: ctx) |> MLIR.is_null()
            end
          end,
          &"#{&2}--#{MLIR.Diagnostic.severity(&1)}--#{to_string(MLIR.location(&1))}--#{to_string(&1)}"
        )

      assert txt == "--error--???:1:1--expected attribute value"
    end
  end
end
