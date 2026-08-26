defmodule Beaver.Shadow.Tuning.GPU do
  @moduledoc """
  Real-kernel latency evaluation for tuning configs.

  Compiles a TTIR source with the config's `num_warps`, translates to PTX,
  loads it through `Beaver.MLIR.CUDA`, allocates the buffers described by
  `launch.buffer_sizes`, and reports the median launch latency across
  `samples`. Buffer placeholder atoms in `launch.args_template` (e.g.
  `{:ptr, :a}`) are resolved to the allocated device pointers.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.GPU

  @type launch() :: %{
          kernel_name: String.t(),
          grid: {pos_integer(), pos_integer(), pos_integer()},
          block: {pos_integer(), pos_integer(), pos_integer()},
          shared_mem: non_neg_integer(),
          samples: pos_integer(),
          buffer_sizes: %{atom() => pos_integer()},
          args_template: [{:ptr | :i64, atom() | integer()}]
        }

  @doc """
  Evaluates one config: lowers `source_text` with `num_warps`, launches on the
  GPU and returns `{:ok, median_ns}` or `{:error, reason}`.
  """
  @spec evaluate(String.t(), MLIR.Context.t(), String.t(), pos_integer(), launch()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def evaluate(source_text, context, target, num_warps, launch) do
    module = MLIR.Module.create!(source_text, ctx: context)

    try do
      llvm =
        Beaver.Triton.compile_to_llvm(module,
          target: target,
          num_warps: num_warps
        )

      ptx = llvm_to_ptx(llvm)
      {:ok, launch_latency_ns(ptx, launch, num_warps)}
    rescue
      exception -> {:error, Exception.message(exception)}
    after
      MLIR.Module.destroy(module)
    end
  end

  defp launch_latency_ns(ptx, launch, num_warps) do
    %{
      kernel_name: kernel_name,
      grid: grid,
      shared_mem: shared_mem,
      samples: samples,
      buffer_sizes: buffer_sizes,
      args_template: args_template
    } = launch

    # The block must match the kernel's launch configuration: one warp per 32
    # threads, so num_warps warps map to 32 * num_warps threads.
    block = {32 * num_warps, 1, 1}

    context = MLIR.Context.create()

    try do
      target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: context)
      module = GPU.binary_module("kernel", target, ptx, format: :assembly, ctx: context)
      {:ok, mod} = MLIR.CUDA.load_gpu_binary(module)
      {:ok, f} = MLIR.CUDA.module_get_function(mod, kernel_name)

      {:ok, buffers} = alloc_buffers(buffer_sizes)

      try do
        resolved_args = resolve_args(args_template, buffers)

        :ok = MLIR.CUDA.launch_kernel(f, grid, block, resolved_args, shared_mem: shared_mem)
        :ok = MLIR.CUDA.synchronize()

        durations =
          for _ <- 1..samples do
            started = System.monotonic_time()
            :ok = MLIR.CUDA.launch_kernel(f, grid, block, resolved_args, shared_mem: shared_mem)
            :ok = MLIR.CUDA.synchronize()
            System.monotonic_time() - started
          end
          |> Enum.sort()

        Enum.at(durations, div(length(durations), 2))
      after
        Enum.each(buffers, fn {_name, ptr} -> MLIR.CUDA.mem_free(ptr) end)
      end
    after
      MLIR.Context.destroy(context)
    end
  end

  defp llvm_to_ptx(llvm) do
    llvm
    |> Beaver.MLIR.Target.LLVMIR.translate!()
    |> Beaver.MLIR.Target.LLVMIR.compile_to_ptx!(cpu: "sm_80")
  end

  defp alloc_buffers(buffer_sizes) do
    Enum.reduce_while(buffer_sizes, {:ok, %{}}, fn {name, size}, {:ok, acc} ->
      case MLIR.CUDA.mem_alloc(size) do
        {:ok, ptr} -> {:cont, {:ok, Map.put(acc, name, ptr)}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  defp resolve_args(args_template, buffers) do
    Enum.map(args_template, fn
      {:ptr, name} when is_atom(name) -> {:ptr, Map.fetch!(buffers, name)}
      other -> other
    end)
  end
end
