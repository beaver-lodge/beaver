defmodule Beaver.Shadow.GPUTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.GPU, as: GPUDialect
  alias Beaver.MLIR.Transform.Schedule.DSL
  alias Beaver.Shadow.{GPU, Receipt, Runner}

  defmodule FakeCUDA do
    def available?, do: true
    def device_count, do: {:ok, 1}
    def device_name(_ordinal), do: {:ok, "Fake GPU"}

    def load_gpu_binary(%MLIR.Module{}), do: {:ok, 100}
    def module_get_function(100, "main"), do: {:ok, 200}
    def mem_alloc(_size), do: {:ok, 300}
    def memcpy_htod(300, _data), do: :ok
    def memcpy_dtoh(300, size), do: {:ok, :binary.copy(<<0>>, size)}
    def launch_kernel(200, {1, 1, 1}, {1, 1, 1}, [_], _opts), do: :ok
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

  @tag skip:
         !System.get_env("BEAVER_CUDA_TEST") or
           System.get_env("BEAVER_TRITON_PREBUILT_DIR") == nil
  @tag :cuda
  test "launches a real Triton matmul and verifies the result", %{ctx: ctx} do
    fixture = Path.expand("fixtures/triton/ttgir_matmul.mlir", __DIR__)
    source = File.read!(fixture)
    context = MLIR.Context.create(all_dialects: false)
    on_exit(fn -> MLIR.Context.destroy(context) end)
    Beaver.Triton.register(context)

    module = MLIR.Module.create!(source, ctx: context)
    on_exit(fn -> MLIR.Module.destroy(module) end)

    llvm = Beaver.Triton.compile_to_llvm(module)

    # compile LLVM IR to PTX through the prebuilt toolchain
    ptx =
      llvm
      |> Beaver.MLIR.Target.LLVMIR.translate!()
      |> Beaver.MLIR.Target.LLVMIR.compile_to_ptx!(cpu: "sm_80")

    assert ptx =~ ".visible .entry matmul_kernel_"

    target = GPUDialect.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

    binary_module =
      GPUDialect.binary_module("matmul", target, ptx, format: :assembly, ctx: ctx)

    on_exit(fn -> MLIR.Module.destroy(binary_module) end)

    {:ok, mod} = MLIR.CUDA.load_gpu_binary(binary_module)
    on_exit(fn -> MLIR.CUDA.module_unload(mod) end)

    {:ok, f} =
      MLIR.CUDA.module_get_function(
        mod,
        "matmul_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32_i32_i32_i32_i32_i32_i32__12c64_13c64_14c64_15c8"
      )

    size = 64 * 64 * 4
    {:ok, d_a} = MLIR.CUDA.mem_alloc(size)
    {:ok, d_b} = MLIR.CUDA.mem_alloc(size)
    {:ok, d_c} = MLIR.CUDA.mem_alloc(size)

    on_exit(fn ->
      MLIR.CUDA.mem_free(d_a)
      MLIR.CUDA.mem_free(d_b)
      MLIR.CUDA.mem_free(d_c)
    end)

    a_data = for _ <- 1..(64 * 64), do: 1.0
    b_data = for _ <- 1..(64 * 64), do: 2.0

    :ok =
      MLIR.CUDA.memcpy_htod(
        d_a,
        a_data |> Enum.map(&<<&1::float-32-little>>) |> IO.iodata_to_binary()
      )

    :ok =
      MLIR.CUDA.memcpy_htod(
        d_b,
        b_data |> Enum.map(&<<&1::float-32-little>>) |> IO.iodata_to_binary()
      )

    :ok = MLIR.CUDA.memcpy_htod(d_c, :binary.copy(<<0::32>>, 64 * 64))

    # row-major strides: A[M,K]=64,1; B[K,N]=64,1; C[M,N]=64,1; M=N=K=64
    args = [
      {:ptr, d_a},
      {:ptr, d_b},
      {:ptr, d_c},
      {:i64, 64},
      {:i64, 64},
      {:i64, 64},
      {:i64, 64},
      {:i64, 1},
      {:i64, 64},
      {:i64, 1},
      {:i64, 64},
      {:i64, 1},
      {:ptr, 0},
      {:ptr, 0}
    ]

    assert :ok = MLIR.CUDA.launch_kernel(f, {1, 1, 1}, {128, 1, 1}, args, shared_mem: 16_384)

    assert {:ok, data} = MLIR.CUDA.memcpy_dtoh(d_c, 64 * 64 * 4)
    floats = for <<f::float-32-little <- data>>, do: f
    assert Enum.all?(floats, &(&1 == 128.0))
  end

  defp own(module) do
    on_exit(fn -> MLIR.Module.destroy(module) end)
    module
  end
end
