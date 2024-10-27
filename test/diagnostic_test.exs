defmodule DiagnosticTest do
  use Beaver.Case, async: true, diagnostic: :server
  alias Beaver.MLIR.Attribute

  defmodule DiagnosticTestHelper do
    def start_and_attach(ctx, cb) do
      {:ok, server} =
        GenServer.start(
          Beaver.DiagnosticsCapturer,
          cb
        )

      handler_id = Beaver.DiagnosticsCapturer.attach(ctx, server)
      {server, handler_id}
    end

    def cleanup_handler(ctx, server, handler_id) do
      :ok = GenServer.stop(server)
      Beaver.MLIR.Diagnostic.detach(ctx, handler_id)
    end

    def get_attr(ctx) do
      Attribute.get("invalid_attr", ctx: ctx)
    end

    def format_with_severity_and_loc(d, acc) do
      "#{acc}--#{MLIR.Diagnostic.severity(d)}--#{to_string(MLIR.location(d))}--#{to_string(d)}"
    end
  end

  @collected "--error--invalid_attr:1:1--expected attribute value"
  @err_msg "fail to parse attribute: invalid_attr"
  describe "server" do
    test "handler", %{ctx: ctx} do
      {server, handler_id} =
        DiagnosticTestHelper.start_and_attach(
          ctx,
          &DiagnosticTestHelper.format_with_severity_and_loc/2
        )

      assert_raise RuntimeError, @err_msg, fn -> DiagnosticTestHelper.get_attr(ctx) end
      assert Beaver.DiagnosticsCapturer.collect(server) == @collected
      DiagnosticTestHelper.cleanup_handler(ctx, server, handler_id)
    end

    test "handler with init state", %{ctx: ctx} do
      {server, handler_id} =
        DiagnosticTestHelper.start_and_attach(
          ctx,
          {fn -> "hello" end, &DiagnosticTestHelper.format_with_severity_and_loc/2}
        )

      assert_raise RuntimeError, @err_msg, fn -> DiagnosticTestHelper.get_attr(ctx) end

      assert Beaver.DiagnosticsCapturer.collect(server) ==
               "hello#{@collected}"

      DiagnosticTestHelper.cleanup_handler(ctx, server, handler_id)
    end
  end

  describe "with_diagnostics" do
    test "no init", %{ctx: ctx} do
      {%RuntimeError{}, txt} =
        Beaver.with_diagnostics(
          ctx,
          fn ->
            assert_raise RuntimeError, @err_msg, fn -> DiagnosticTestHelper.get_attr(ctx) end
          end,
          &DiagnosticTestHelper.format_with_severity_and_loc/2
        )

      assert txt == @collected
    end
  end
end
