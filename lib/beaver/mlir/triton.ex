defmodule Beaver.Triton do
  @moduledoc """
  Registers Triton's core dialects and passes in a Beaver MLIR context.

  Once registered, a context can parse and inspect Triton IR (`tt`, `ttgir`,
  `ttng`, `ttinstrument`, `gluon` dialects), and Triton passes can be driven
  by name through `Beaver.Composer` pipelines.

  Requires a Beaver native build linked against the Triton core prebuilt
  (build with `BEAVER_TRITON_PREBUILT_DIR` set); otherwise the calls raise.
  """

  alias Beaver.MLIR

  @doc """
  Registers Triton passes and dialects, then loads the dialects on `context`.
  """
  @spec register(MLIR.Context.t()) :: :ok
  def register(%MLIR.Context{} = context) do
    register_passes()

    unless MLIR.CAPI.beaver_raw_triton_register_dialects(context.ref) do
      raise ArgumentError,
            "Beaver was built without Triton support; rebuild the native " <>
              "library with BEAVER_TRITON_PREBUILT_DIR set"
    end

    :ok
  end

  @doc """
  Registers Triton's passes in the global MLIR pass registry.

  Idempotent per process; safe to call multiple times.
  """
  @spec register_passes() :: :ok
  def register_passes do
    unless :persistent_term.get({__MODULE__, :passes_registered}, false) do
      unless MLIR.CAPI.beaver_raw_triton_register_passes() do
        raise ArgumentError,
              "Beaver was built without Triton support; rebuild the native " <>
                "library with BEAVER_TRITON_PREBUILT_DIR set"
      end

      :persistent_term.put({__MODULE__, :passes_registered}, true)
    end

    :ok
  end

  @doc """
  Lowers a TTIR module to LLVM IR through the full NVIDIA Triton pipeline.

  This mirrors Triton's `make_ttgir` + `make_llir` pass chains for a CUDA
  target. Running `convert-triton-gpu-to-llvm` alone is not sufficient for
  real kernels: shared-memory allocation, warp-group allocation, SCF-to-CF,
  warp-specialize lowering, and NVGPU lowering must run first, otherwise
  kernels with layout conversions either fail verification or crash the
  process.

  Requires a context with Triton dialects registered (`register/1`).
  """
  @spec compile_to_llvm(MLIR.Module.t(), keyword()) :: MLIR.Module.t()
  def compile_to_llvm(%MLIR.Module{} = module, opts \\ []) do
    target = Keyword.get(opts, :target, "cuda:80")
    num_warps = Keyword.get(opts, :num_warps, 4)
    remove_layouts? = Keyword.get(opts, :remove_layout_conversions, true)
    from_ttgir? = Keyword.get(opts, :from_ttgir, false)
    # `convert-triton-gpu-to-llvm` (and the NV passes with the same options)
    # take compute-capability/ptx-version as pass options; driven by name they
    # default to 80, which breaks fp8 lowering on sm_89+ targets. Derive them
    # from the target string (cuda:120 -> 120).
    capability =
      case Regex.run(~r/cuda:(\d+)/, target) do
        [_, cc] -> String.to_integer(cc)
        _ -> 80
      end

    ttgpu_pipeline =
      [
        unless(from_ttgir?,
          do: "convert-triton-to-tritongpu{target=#{target}, num-warps=#{num_warps}}"
        ),
        "tritongpu-coalesce",
        "tritongpu-F32DotTC",
        "triton-nvidia-gpu-plan-cta",
        if(remove_layouts?, do: "tritongpu-remove-layout-conversions"),
        "tritongpu-optimize-thread-locality",
        "tritongpu-accelerate-matmul",
        if(remove_layouts?, do: "tritongpu-remove-layout-conversions"),
        "tritongpu-optimize-dot-operands",
        "canonicalize"
      ]
      |> Enum.reject(&is_nil/1)

    Enum.reduce(ttgpu_pipeline, module, fn pass, acc ->
      Beaver.Composer.append(acc, pass)
    end)
    |> then(fn composer ->
      composer
      # make_llir: TritonGPU -> LLVM
      |> Beaver.Composer.append("tritongpu-combine-tensor-select-and-if")
      |> Beaver.Composer.append("tritongpu-allocate-warp-groups")
      |> Beaver.Composer.append("convert-scf-to-cf")
      |> Beaver.Composer.append(
        "allocate-shared-memory-nv{compute-capability=#{capability} ptx-version=#{capability}}"
      )
      |> Beaver.Composer.append("triton-tensor-memory-allocation")
      |> Beaver.Composer.append("triton-nvidia-check-matmul-two-cta")
      |> Beaver.Composer.append("triton-nvidia-gpu-proxy-fence-insertion")
      |> Beaver.Composer.append("triton-nvidia-gpu-tmem-barrier-insertion")
      |> Beaver.Composer.append(
        "convert-triton-gpu-to-llvm{compute-capability=#{capability} ptx-version=#{capability}}"
      )
      |> Beaver.Composer.append("convert-warp-specialize-to-llvm")
      |> Beaver.Composer.append("convert-nv-gpu-to-llvm")
      |> Beaver.Composer.append("convert-nvvm-to-llvm")
      |> Beaver.Composer.append("canonicalize")
    end)
    |> Beaver.Composer.run!()
  end
end
