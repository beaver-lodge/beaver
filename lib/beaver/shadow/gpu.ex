defmodule Beaver.Shadow.GPU do
  @moduledoc """
  GPU-backed evaluator for `Beaver.Shadow.Runner`.

  This evaluator closes the Shadow Wavefront loop on real hardware:

  ```text
  payload source → package_binary! (NVVM/ROCDL) → CUDA.load_gpu_binary
  → module_get_function → mem_alloc/memcpy_htod → launch_kernel
  → memcpy_dtoh → receipt metadata
  ```

  It implements the same evaluator contract as `Runner`'s default surrogate:
  `(resolved, candidate) -> {:ok, value, metadata} | {:error, reason}`.
  Measurements (native-time durations, kernel launch evidence) go into
  `metadata` and never into `Receipt.identity/1`.

  On machines without a CUDA driver the evaluator does not crash: it returns
  `{:error, {:cuda_unavailable, reason}}` so the experiment loop degrades to a
  recorded failure with a distinguishable category.

  The backend module is injectable for tests (`:backend` option); it defaults
  to `Beaver.MLIR.CUDA`.
  """

  alias Beaver.MLIR
  alias MLIR.Dialect.GPU, as: GPUDialect
  alias Beaver.Shadow.Runner
  alias Beaver.Shadow.GPU.ArtifactCache
  alias MLIR.Transform.Schedule

  @type backend() :: module()

  @doc """
  Runs one GPU experiment over all candidates of `schedule` against `source`.

  Options are forwarded to `Runner.run/3` plus:

    * `:target` — NVVM or ROCDL target attribute (defaults to `sm_80` NVVM)
    * `:backend` — CUDA backend module (defaults to `Beaver.MLIR.CUDA`)
    * `:grid` / `:block` — launch geometry tuples (defaults to `{1, 1, 1}`)
    * `:kernel_name` — kernel to launch inside the packaged binary
    * `:cache` — a `CompilationCache` for packaged GPU binaries; when set, the
      `gpu-module-to-binary` compilation is reused for repeated `(source, target)`
      pairs and reported as `:hit`/`:miss` in the receipt
  """
  @spec run(binary(), Schedule.input(), keyword()) :: Runner.Run.t()
  def run(source, schedule, opts \\ []) do
    evaluator = &evaluate(&1, &2, opts)
    Runner.run(source, schedule, Keyword.put(opts, :evaluator, evaluator))
  end

  @doc """
  Evaluator implementation for `Runner`.

  `opts` must provide `:source` (the payload to package and launch). The
  remaining options match `run/3`.
  """
  @spec evaluate(Schedule.Resolved.t(), %{index: non_neg_integer(), choices: map()}, keyword()) ::
          {:ok, term(), map()} | {:error, term()}
  def evaluate(resolved, candidate, opts) do
    source = Keyword.fetch!(opts, :source)
    backend = Keyword.get(opts, :backend, MLIR.CUDA)

    case backend.available?() do
      false ->
        {:error, {:cuda_unavailable, "no loadable CUDA driver (libcuda)"}}

      true ->
        evaluate_with_cuda(source, resolved, candidate, backend, opts)
    end
  end

  defp evaluate_with_cuda(source, resolved, candidate, backend, opts) do
    context = MLIR.Context.create()
    owned_module? = not match?(%MLIR.Module{}, source)

    try do
      module = compile_source!(source, context)

      try do
        package_and_launch(module, resolved, candidate, backend, opts)
      after
        if owned_module?, do: MLIR.Module.destroy(module)
      end
    after
      MLIR.Context.destroy(context)
    end
  end

  defp compile_source!(%MLIR.Module{} = module, _context), do: module

  defp compile_source!(source, context) when is_binary(source) do
    MLIR.Module.create!(source, ctx: context)
  end

  defp package_and_launch(module, resolved, candidate, backend, opts) do
    target =
      Keyword.get(
        opts,
        :target,
        GPUDialect.nvvm_target_attribute(chip: "sm_80", ctx: MLIR.context(module))
      )

    grid = Keyword.get(opts, :grid, {1, 1, 1})
    block = Keyword.get(opts, :block, {1, 1, 1})
    kernel_name = Keyword.get(opts, :kernel_name, "main")

    started = System.monotonic_time()

    source_key = MLIR.Bytecode.write!(module)
    {packaged, cache_status} = packaged_binary(module, source_key, target, opts)

    result =
      try do
        with {:ok, module_handle} <- backend.load_gpu_binary(packaged),
             {:ok, function_handle} <- backend.module_get_function(module_handle, kernel_name) do
          launch_result =
            launch_once(backend, function_handle, grid, block, module_handle, opts)

          duration = System.monotonic_time() - started

          case launch_result do
            {:ok, measurements} ->
              {:ok, measurements,
               %{
                 artifact: %{
                   cache: cache_status,
                   lookup_key: resolved.digest,
                   artifact_key: ArtifactCache.key(source_key, target)
                 },
                 trace: %{
                   action_count: 1,
                   tags: ["gpu.launch"],
                   candidate_index: candidate.index,
                   schedule_digest: resolved.digest
                 },
                 device: device_facts(backend),
                 durations: Map.put(measurements, :total_native, duration)
               }}

            {:error, reason} ->
              {:error, {:launch_failure, reason}}
          end
        else
          {:error, reason} -> {:error, {:cuda_load_failure, reason}}
        end
      after
        # The packaged module is a separate object from `module`.
        if packaged != module, do: MLIR.Module.destroy(packaged)
      end

    result
  end

  defp packaged_binary(module, source_key, target, opts) do
    case Keyword.get(opts, :cache) do
      nil ->
        {GPUDialect.package_binary!(module, target, format: Keyword.get(opts, :format, :isa)),
         :miss}

      cache ->
        case ArtifactCache.get(cache, source_key, target, MLIR.context(module)) do
          {:ok, packaged} ->
            {packaged, :hit}

          :miss ->
            packaged =
              GPUDialect.package_binary!(module, target, format: Keyword.get(opts, :format, :isa))

            ArtifactCache.put(cache, source_key, target, packaged)
            {packaged, :miss}

          {:error, reason} ->
            raise ArgumentError, "GPU artifact cache lookup failed: #{inspect(reason)}"
        end
    end
  end

  # Launch with a small round-trip payload: allocate 4 bytes, copy in/out, and
  # launch the kernel with one pointer argument. Kernels that do not expect a
  # pointer argument can pass `:arg_count`/`:skip_io` options instead.
  defp launch_once(backend, function_handle, grid, block, module_handle, opts) do
    if Keyword.get(opts, :skip_io, false) do
      case backend.launch_kernel(function_handle, grid, block, []) do
        :ok -> {:ok, %{launch_native: System.monotonic_time()}}
        {:error, reason} -> {:error, reason}
      end
    else
      with {:ok, device_ptr} <- backend.mem_alloc(4),
           :ok <- backend.memcpy_htod(device_ptr, <<0, 0, 0, 0>>) do
        try do
          case backend.launch_kernel(function_handle, grid, block, [{:ptr, device_ptr}]) do
            :ok ->
              case backend.memcpy_dtoh(device_ptr, 4) do
                {:ok, _data} ->
                  {:ok,
                   %{
                     launch_native: System.monotonic_time(),
                     bytes: 4
                   }}

                {:error, reason} ->
                  {:error, reason}
              end

            {:error, reason} ->
              {:error, reason}
          end
        after
          backend.mem_free(device_ptr)
        end
      end
    end
  end

  defp device_facts(backend) do
    with {:ok, count} <- backend.device_count(),
         {:ok, name} <- backend.device_name(0) do
      %{device_count: count, device_name: name}
    else
      {:error, reason} -> %{device_error: reason}
    end
  end
end
