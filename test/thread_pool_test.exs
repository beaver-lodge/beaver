defmodule ThreadPoolTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR

  test "multiple contexts share an owned pool and defer teardown" do
    {:ok, owner} = MLIR.ThreadPool.start_link()
    monitor = Process.monitor(owner)

    first = MLIR.Context.create(thread_pool: owner)
    second = MLIR.Context.create(thread_pool: owner)

    assert first.thread_pool_owner == owner
    assert second.thread_pool_owner == owner

    first_threads =
      MLIR.CAPI.mlirContextGetNumThreads(first)
      |> Beaver.Native.to_term()

    second_threads =
      MLIR.CAPI.mlirContextGetNumThreads(second)
      |> Beaver.Native.to_term()

    assert first_threads == MLIR.ThreadPool.max_concurrency(owner)
    assert second_threads == first_threads
    assert :deferred = MLIR.ThreadPool.close(owner)

    assert_raise ArgumentError, ~r/thread pool is closing/, fn ->
      MLIR.Context.create(thread_pool: owner)
    end

    MLIR.Context.destroy(first)
    assert Process.alive?(owner)
    MLIR.Context.destroy(second)

    assert_receive {:DOWN, ^monitor, :process, ^owner, :normal}
  end

  test "context threading and registry options are explicit" do
    context = MLIR.Context.create(threading: false, all_dialects: false)

    assert MLIR.CAPI.mlirContextGetNumThreads(context) |> Beaver.Native.to_term() == 1
    MLIR.Context.destroy(context)

    registry = MLIR.CAPI.mlirDialectRegistryCreate()
    MLIR.CAPI.mlirRegisterAllDialects(registry)

    context =
      MLIR.Context.create(
        registry: registry,
        all_dialects: false,
        thread_pool: nil
      )

    assert MLIR.CAPI.mlirContextGetNumRegisteredDialects(context) |> Beaver.Native.to_term() > 0

    MLIR.Context.destroy(context)
    MLIR.CAPI.mlirDialectRegistryDestroy(registry)
  end

  test "a disabled context rejects an external pool" do
    assert_raise ArgumentError, ~r/thread_pool/, fn ->
      MLIR.Context.create(threading: false, thread_pool: :application)
    end
  end
end
