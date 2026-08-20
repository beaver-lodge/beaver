defmodule ContextTransientScopeTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  test "reports the linked LLVM capability", %{ctx: ctx} do
    if MLIR.Context.transient_scope_supported?() do
      assert :ok ==
               MLIR.Context.with_transient_scope(ctx, fn scoped_ctx ->
                 module = MLIR.Module.create!("module {}", ctx: scoped_ctx)
                 MLIR.Module.destroy(module)
                 :ok
               end)
    else
      assert_raise ArgumentError, ~r/does not support transient context scopes/, fn ->
        MLIR.Context.with_transient_scope(ctx, fn _ctx -> :ok end)
      end
    end
  end

  test "always ends a supported scope after a callback failure", %{ctx: ctx} do
    if MLIR.Context.transient_scope_supported?() do
      assert_raise RuntimeError, "boom", fn ->
        MLIR.Context.with_transient_scope(ctx, fn _ctx -> raise "boom" end)
      end

      assert :reentered ==
               MLIR.Context.with_transient_scope(ctx, fn _ctx -> :reentered end)
    end
  end

  test "rejects concurrent use of the same supported context", %{ctx: ctx} do
    if MLIR.Context.transient_scope_supported?() do
      parent = self()

      task =
        Task.async(fn ->
          MLIR.Context.with_transient_scope(ctx, fn _ctx ->
            send(parent, :entered)
            receive do: (:leave -> :ok)
          end)
        end)

      assert_receive :entered

      assert_raise ArgumentError, ~r/already has an active transient scope/, fn ->
        MLIR.Context.with_transient_scope(ctx, fn _ctx -> :ok end)
      end

      assert_raise ArgumentError,
                   ~r/cannot destroy a context with an active transient scope/,
                   fn ->
                     MLIR.Context.destroy(ctx)
                   end

      send(task.pid, :leave)
      assert Task.await(task) == :ok
    end
  end
end
