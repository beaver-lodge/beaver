defmodule Beaver.Shadow.GPUTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.Shadow.{GPU, Receipt, Runner}
  alias Beaver.MLIR.Transform.Schedule.DSL

  defmodule FakeCUDA do
    def available?, do: true
    def device_count, do: {:ok, 1}
    def device_name(_ordinal), do: {:ok, "Fake GPU"}

    def load_gpu_binary(%MLIR.Module{}), do: {:ok, 100}
    def module_get_function(100, "main"), do: {:ok, 200}
    def mem_alloc(_size), do: {:ok, 300}
    def memcpy_htod(300, _data), do: :ok
    def memcpy_dtoh(300, size), do: {:ok, :binary.copy(<<0>>, size)}
    def launch_kernel(200, {1, 1, 1}, {1, 1, 1}, [_]), do: :ok
    def mem_free(300), do: :ok
    def module_unload(_), do: :ok
  end

  defmodule NoCUDA do
    def available?, do: false
  end

  defmodule ClosedLoopSchedule do
    use DSL

    defschedule closed_loop do
      sequence "__transform_main", [_root >>> any_op()] do
        alternatives "route" do
          branch do
            _fast = knob("fast_tile", [8, 16])
          end

          branch do
            _slow = knob("slow_tile", [32, 64])
          end
        end
      end
    end
  end

  @payload """
  module {
    gpu.module @kernels {
      gpu.func @main() kernel {
        gpu.return
      }
    }
  }
  """

  setup do
    context = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(context) end)
    %{ctx: context}
  end

  test "degrades to a recorded failure without a CUDA driver", %{ctx: ctx} do
    schedule = own(ClosedLoopSchedule.closed_loop(ctx: ctx))

    run =
      GPU.run(@payload, schedule,
        backend: NoCUDA,
        source: @payload
      )

    assert Enum.all?(run.receipts, &(&1.status == :failed))
    assert Enum.all?(run.receipts, &match?(%{kind: :evaluation_failure}, &1.failure))
    assert Enum.all?(run.receipts, &match?({:cuda_unavailable, _}, &1.failure.reason))
    assert run.winner == nil
  end

  test "records launch evidence through a fake backend", %{ctx: ctx} do
    schedule = own(ClosedLoopSchedule.closed_loop(ctx: ctx))

    run =
      GPU.run(@payload, schedule,
        backend: FakeCUDA,
        source: @payload
      )

    assert length(run.receipts) == 4
    assert Enum.all?(run.receipts, &(&1.status == :ok))

    winner = run.winner
    assert winner.trace.tags == ["gpu.launch"]
    assert winner.user_metadata.device == %{device_count: 1, device_name: "Fake GPU"}
    assert is_integer(winner.user_metadata.durations.total_native)

    assert Receipt.identity(winner) ==
             winner |> Receipt.encode!() |> Receipt.decode!() |> Receipt.identity()
  end

  test "reuses packaged GPU binaries through the artifact cache", %{ctx: ctx} do
    cache = start_supervised!({MLIR.CompilationCache.Memory, []})
    schedule = own(ClosedLoopSchedule.closed_loop(ctx: ctx))

    first =
      GPU.run(@payload, schedule,
        backend: FakeCUDA,
        source: @payload,
        cache: {:memory, cache}
      )

    second =
      GPU.run(@payload, schedule,
        backend: FakeCUDA,
        source: @payload,
        cache: {:memory, cache}
      )

    assert Enum.map(first.receipts, & &1.artifact.cache) == [:miss, :hit, :hit, :hit]
    assert Enum.map(second.receipts, & &1.artifact.cache) == [:hit, :hit, :hit, :hit]

    assert Receipt.identity(hd(first.receipts)) == Receipt.identity(hd(second.receipts))
  end

  @tag skip: !System.get_env("BEAVER_CUDA_TEST")
  @tag :cuda
  test "launches a real kernel end to end on a CUDA machine", %{ctx: ctx} do
    schedule = own(ClosedLoopSchedule.closed_loop(ctx: ctx))

    run =
      GPU.run(@payload, schedule,
        backend: MLIR.CUDA,
        source: @payload
      )

    assert %Runner.Run{winner: %Receipt{status: :ok}} = run
    assert run.winner.user_metadata.device.device_count >= 1
  end

  defp own(module) do
    on_exit(fn -> MLIR.Module.destroy(module) end)
    module
  end
end
