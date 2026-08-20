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

  test "collects stale non-owning wrappers safely after a supported scope ends", %{ctx: ctx} do
    if MLIR.Context.transient_scope_supported?() do
      parent = self()

      {pid, monitor} =
        spawn_monitor(fn ->
          stale_wrappers =
            MLIR.Context.with_transient_scope(ctx, fn scoped_ctx ->
              types =
                Enum.map([8, 16, 32], fn bitwidth ->
                  MLIR.Type.integer(bitwidth, ctx: scoped_ctx)
                end)

              attributes = Enum.map(types, &MLIR.Attribute.type/1)

              types
              |> Enum.zip(attributes)
              |> Enum.flat_map(fn {type, attribute} -> [attribute, type] end)
              |> Enum.reverse()
            end)

          drop_and_collect(parent, length(stale_wrappers))
        end)

      assert_receive {:transient_wrappers_collected, ^pid, 6}
      assert_receive {:DOWN, ^monitor, :process, ^pid, :normal}

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

  defp drop_and_collect(parent, wrapper_count) do
    :erlang.garbage_collect()
    send(parent, {:transient_wrappers_collected, self(), wrapper_count})
  end
end
