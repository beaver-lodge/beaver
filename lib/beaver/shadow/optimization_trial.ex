defmodule Beaver.Shadow.OptimizationTrial do
  @moduledoc """
  A reproducible compiler-side optimization trial on real Triton IR.

  The trial answers one question: does a layout strategy reduce the number of
  `ttg.convert_layout` operations that survive in the TTGIR, and by how much?
  It is the first Shadow Wavefront consumer that compares two real pipeline
  variants on a real workload instead of a synthetic fixture.

  The baseline strategy runs `convert-triton-to-tritongpu` only. The optimized
  strategy additionally runs `tritongpu-remove-layout-conversions`, which
  propagates layouts through the graph and eliminates redundant conversions.
  Both variants are audited with `Beaver.MLIR.Triton.LayoutAudit`, so the
  reduction is a structural fact, not a subjective reading of the IR.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.GPU

  defmodule Result do
    @moduledoc "The audited outcome of one trial."
    @enforce_keys [:input_digest, :baseline, :optimized, :reduced]
    defstruct [
      :input_digest,
      :baseline,
      :optimized,
      :reduced,
      :lowered_to_llvm,
      :gpu_baseline_ns,
      :gpu_optimized_ns,
      :gpu_speedup
    ]

    @type t() :: %__MODULE__{
            input_digest: String.t(),
            baseline: non_neg_integer(),
            optimized: non_neg_integer(),
            reduced: boolean(),
            lowered_to_llvm: boolean(),
            gpu_baseline_ns: non_neg_integer() | nil,
            gpu_optimized_ns: non_neg_integer() | nil,
            gpu_speedup: float() | nil
          }
  end

  @doc """
  Runs the layout trial on a real Triton IR module.

  Options:

    * `:target` — Triton GPU target, defaults to `cuda:80`
  """
  @spec run(MLIR.Module.t(), keyword()) :: Result.t()
  def run(%MLIR.Module{} = module, opts \\ []) do
    target = Keyword.get(opts, :target, "cuda:80")
    num_warps = Keyword.get(opts, :num_warps, 4)
    source_text = MLIR.to_string(module)
    context = MLIR.context(module)
    gpu? = Keyword.get(opts, :gpu, false)

    baseline =
      module
      |> Beaver.Composer.append(
        "convert-triton-to-tritongpu{target=#{target}, num-warps=#{num_warps}}"
      )
      |> Beaver.Composer.run!()

    baseline_count = audit_count(baseline)

    optimized =
      baseline
      |> Beaver.Composer.append("tritongpu-remove-layout-conversions")
      |> Beaver.Composer.run!()

    optimized_count = audit_count(optimized)

    lowered_to_llvm =
      try do
        fresh = MLIR.Module.create!(source_text, ctx: context)

        # `compile_to_llvm` mutates and returns the same module (`fresh`), so
        # the text must be read before the module is destroyed.
        result =
          try do
            llvm = Beaver.Triton.compile_to_llvm(fresh, target: target)
            MLIR.to_string(llvm) =~ "llvm.func"
          after
            MLIR.Module.destroy(fresh)
          end

        result
      rescue
        exception ->
          IO.warn("compile_to_llvm failed: #{Exception.message(exception)}")
          false
      end

    gpu_results =
      if gpu? and lowered_to_llvm do
        gpu_latency_compare(source_text, context, target, opts)
      else
        %{baseline_ns: nil, optimized_ns: nil, speedup: nil}
      end

    %Result{
      input_digest:
        module
        |> MLIR.Bytecode.write!()
        |> then(&:crypto.hash(:sha256, &1))
        |> Base.encode16(case: :lower),
      baseline: baseline_count,
      optimized: optimized_count,
      reduced: optimized_count < baseline_count,
      lowered_to_llvm: lowered_to_llvm,
      gpu_baseline_ns: gpu_results.baseline_ns,
      gpu_optimized_ns: gpu_results.optimized_ns,
      gpu_speedup: gpu_results.speedup
    }
  end

  @doc """
  Compares real GPU latency between the baseline and optimized pipelines.

  Both variants are compiled independently from the same TTIR source (baseline
  skips `tritongpu-remove-layout-conversions`, optimized runs it), translated
  to PTX, loaded through `Beaver.MLIR.CUDA`, and launched several times. The
  median launch time in nanoseconds is reported for each variant.
  """
  @spec gpu_latency_compare(String.t(), MLIR.Context.t(), String.t(), keyword()) :: map()
  def gpu_latency_compare(source_text, context, target, opts) do
    kernel_name = Keyword.get(opts, :kernel_name, matmul_kernel_name())
    grid = Keyword.get(opts, :grid, {1, 1, 1})
    block = Keyword.get(opts, :block, {128, 1, 1})
    shared_mem = Keyword.get(opts, :shared_mem, 16_384)
    args = Keyword.get(opts, :args)
    samples = Keyword.get(opts, :samples, 10)

    launch = %{
      kernel_name: kernel_name,
      grid: grid,
      block: block,
      shared_mem: shared_mem,
      args: args,
      samples: samples
    }

    baseline_ns =
      source_text
      |> compile_and_launch(context, target, launch, remove_layout_conversions: false)

    optimized_ns =
      source_text
      |> compile_and_launch(context, target, launch, remove_layout_conversions: true)

    %{
      baseline_ns: baseline_ns,
      optimized_ns: optimized_ns,
      speedup: speedup(baseline_ns, optimized_ns)
    }
  end

  defp compile_and_launch(source_text, context, target, launch, compile_opts) do
    module = MLIR.Module.create!(source_text, ctx: context)

    result =
      try do
        llvm = Beaver.Triton.compile_to_llvm(module, Keyword.put(compile_opts, :target, target))
        ptx = llvm_to_ptx(llvm)
        launch_latency_ns(ptx, launch)
      rescue
        exception ->
          IO.warn("compile_and_launch failed: #{Exception.message(exception)}")
          nil
      end

    MLIR.Module.destroy(module)
    result
  end

  defp llvm_to_ptx(llvm) do
    llvm
    |> Beaver.MLIR.Target.LLVMIR.translate!()
    |> Beaver.MLIR.Target.LLVMIR.compile_to_ptx!(cpu: "sm_80")
  end

  defp launch_latency_ns(ptx, launch) do
    %{
      kernel_name: kernel_name,
      grid: grid,
      block: block,
      shared_mem: shared_mem,
      args: args,
      samples: samples
    } = launch

    context = MLIR.Context.create()

    try do
      target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: context)
      module = GPU.binary_module("kernel", target, ptx, format: :assembly, ctx: context)
      {:ok, mod} = MLIR.CUDA.load_gpu_binary(module)
      {:ok, f} = MLIR.CUDA.module_get_function(mod, kernel_name)

      durations =
        with_buffers(args, fn resolved_args ->
          # Warm up once so the first launch's lazy loading is not measured.
          :ok = MLIR.CUDA.launch_kernel(f, grid, block, resolved_args, shared_mem: shared_mem)
          :ok = MLIR.CUDA.synchronize()

          for _ <- 1..samples do
            started = System.monotonic_time()
            :ok = MLIR.CUDA.launch_kernel(f, grid, block, resolved_args, shared_mem: shared_mem)
            :ok = MLIR.CUDA.synchronize()
            System.monotonic_time() - started
          end
          |> Enum.sort()
        end)

      MLIR.CUDA.module_unload(mod)
      MLIR.Module.destroy(module)
      Enum.at(durations, div(length(durations), 2))
    after
      MLIR.Context.destroy(context)
    end
  end

  defp with_buffers(nil, fun) do
    size = 64 * 64 * 4
    {:ok, d_a} = MLIR.CUDA.mem_alloc(size)
    {:ok, d_b} = MLIR.CUDA.mem_alloc(size)
    {:ok, d_c} = MLIR.CUDA.mem_alloc(size)

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

    result = fun.(args)
    MLIR.CUDA.mem_free(d_a)
    MLIR.CUDA.mem_free(d_b)
    MLIR.CUDA.mem_free(d_c)
    result
  end

  defp with_buffers(args, fun) when is_list(args), do: fun.(args)

  defp speedup(nil, _), do: nil
  defp speedup(_, nil), do: nil
  defp speedup(baseline, optimized) when baseline > 0, do: baseline / optimized

  defp matmul_kernel_name do
    "matmul_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32_i32_i32_i32_i32_i32_i32__12c64_13c64_14c64_15c8"
  end

  defp audit_count(module) do
    module
    |> MLIR.Triton.LayoutAudit.audit()
    |> Map.fetch!(:operation_count)
  end
end
