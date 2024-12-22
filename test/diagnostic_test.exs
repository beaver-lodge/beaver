defmodule DiagnosticTest do
  use Beaver.Case, async: true, diagnostic: :server
  alias Beaver.MLIR
  use Beaver
  alias MLIR.{Type, Attribute}
  alias MLIR.Dialect.{Func, Builtin}
  require Func

  defmodule DiagnosticTestHelper do
    def start_and_attach(ctx, cb) do
      {:ok, server} =
        GenServer.start(
          Beaver.Capturer,
          cb
        )

      handler_id = Beaver.Capturer.attach(ctx, server)
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
      assert Beaver.Capturer.collect(server) == @collected
      DiagnosticTestHelper.cleanup_handler(ctx, server, handler_id)
    end

    test "handler with init state", %{ctx: ctx} do
      {server, handler_id} =
        DiagnosticTestHelper.start_and_attach(
          ctx,
          {fn -> "hello" end, &DiagnosticTestHelper.format_with_severity_and_loc/2}
        )

      assert_raise RuntimeError, @err_msg, fn -> DiagnosticTestHelper.get_attr(ctx) end

      assert Beaver.Capturer.collect(server) ==
               "hello#{@collected}"

      DiagnosticTestHelper.cleanup_handler(ctx, server, handler_id)
    end
  end

  describe "with_diagnostics" do
    test "no init", %{ctx: ctx} do
      {%RuntimeError{}, txt} =
        MLIR.Context.with_diagnostics(
          ctx,
          fn ->
            assert_raise RuntimeError, @err_msg, fn -> DiagnosticTestHelper.get_attr(ctx) end
          end,
          &DiagnosticTestHelper.format_with_severity_and_loc/2
        )

      assert txt == @collected
    end

    defp unrealized_conversion_cast_f(ctx) do
      import MLIR.Conversion

      mlir ctx: ctx do
        module do
          Func.func some_func(
                      function_type: Type.function([Type.i64()], [Type.i32()]),
                      sym_name: MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                    ) do
            region do
              block _(a >>> Type.i64()) do
                v0 = Builtin.unrealized_conversion_cast(a) >>> Type.i32()
                Func.return(v0) >>> []
              end
            end
          end
        end
      end
      |> convert_func_to_llvm()
      |> Beaver.Composer.run!()
      |> MLIR.ExecutionEngine.create!(dirty: :io_bound)
    end

    test "large invalid llvm ir", %{ctx: ctx} do
      {_, d_str} =
        MLIR.Context.with_diagnostics(
          ctx,
          fn ->
            assert_raise RuntimeError, fn -> unrealized_conversion_cast_f(ctx) end
          end,
          fn d, _acc -> MLIR.to_string(d) end
        )

      assert d_str =~ "LLVM Translation failed for operation: builtin.unrealized_conversion_cast"
    end
  end
end
