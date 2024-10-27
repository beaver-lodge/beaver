defmodule DiagnosticTest do
  use Beaver.Case, async: true, diagnostic: :server
  alias Beaver.MLIR.Attribute

  defmodule DiagnosticTestHelper do
    def start_and_attach(ctx, cb) do
      {:ok, server} =
        GenServer.start(
          Beaver.DiagnosticHandlerRunner,
          cb
        )

      handler_id = Beaver.DiagnosticHandlerRunner.attach(ctx, server)
      {server, handler_id}
    end

    def cleanup_handler(ctx, server, handler_id) do
      :ok = GenServer.stop(server)
      Beaver.MLIR.Diagnostic.detach(ctx, handler_id)
    end

    def format_with_severity_and_loc(d, acc) do
      "#{acc}--#{MLIR.Diagnostic.severity(d)}--#{to_string(MLIR.location(d))}--#{to_string(d)}"
    end
  end

  describe "server" do
    test "handler", %{ctx: ctx} do
      {server, handler_id} =
        DiagnosticTestHelper.start_and_attach(
          ctx,
          &DiagnosticTestHelper.format_with_severity_and_loc/2
        )

      assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
        Attribute.get("???", ctx: ctx)
      end

      assert Beaver.DiagnosticHandlerRunner.collect(server) ==
               "--error--???:1:1--expected attribute value"

      DiagnosticTestHelper.cleanup_handler(ctx, server, handler_id)
    end

    test "handler with init state", %{ctx: ctx} do
      {server, handler_id} =
        DiagnosticTestHelper.start_and_attach(
          ctx,
          {fn -> "hello" end, &DiagnosticTestHelper.format_with_severity_and_loc/2}
        )

      assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
        Attribute.get("???", ctx: ctx)
      end

      assert Beaver.DiagnosticHandlerRunner.collect(server) ==
               "hello--error--???:1:1--expected attribute value"

      DiagnosticTestHelper.cleanup_handler(ctx, server, handler_id)
    end
  end

  describe "with_diagnostics" do
    test "no init", %{ctx: ctx} do
      {%RuntimeError{}, txt} =
        Beaver.with_diagnostics(
          ctx,
          fn ->
            assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
              Attribute.get("???", ctx: ctx)
            end
          end,
          &DiagnosticTestHelper.format_with_severity_and_loc/2
        )

      assert txt == "--error--???:1:1--expected attribute value"
    end
  end
end
