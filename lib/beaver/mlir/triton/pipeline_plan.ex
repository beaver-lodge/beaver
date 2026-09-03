defmodule Beaver.MLIR.Triton.PipelinePlan do
  @moduledoc """
  Pure, auditable pass plans for Beaver's NVIDIA Triton lowering.

  `:legacy` preserves Beaver's historical lowering and remains the default.
  `:pinned_nvidia_sm100` freezes the `make_ttgir` order used by the pinned
  Triton prebuilt for CUDA capabilities 100 and newer.  The frozen profile is
  deliberately opt-in until a downstream GPU qualification admits it.

  This module contains data only.  In particular, it never imports or parses
  Triton's Python compiler at runtime.
  """

  @schema "beaver.triton.pipeline-plan.v1"
  @default_target "cuda:80"
  @default_num_warps 4
  @default_num_stages 3

  @pinned_provenance %{
    triton_identity: "7d0817d08273d205",
    source_path: "third_party/nvidia/backend/compiler.py",
    source_symbol: "CUDABackend.make_ttgir",
    source_sha256: "d2b6c6b79146143fa56107133ddceec6369812d29e74cb362a7d400476765be8"
  }

  @enforce_keys [
    :schema,
    :profile,
    :target,
    :capability,
    :num_warps,
    :num_stages,
    :provenance,
    :passes
  ]
  defstruct @enforce_keys

  @type profile :: :legacy | :pinned_nvidia_sm100
  @type phase :: :ttgir | :llir
  @type pass :: %{id: atom(), phase: phase(), pipeline: String.t()}
  @type t :: %__MODULE__{
          schema: String.t(),
          profile: profile(),
          target: String.t(),
          capability: pos_integer(),
          num_warps: pos_integer(),
          num_stages: pos_integer(),
          provenance: map() | nil,
          passes: [pass()]
        }

  @doc "Returns the immutable provenance attached to the pinned profile."
  @spec pinned_provenance() :: map()
  def pinned_provenance, do: @pinned_provenance

  @doc "Builds a deterministic lowering plan without touching MLIR state."
  @spec build(keyword()) :: t()
  def build(opts \\ []) when is_list(opts) do
    profile = Keyword.get(opts, :pipeline_profile, :legacy)
    target = Keyword.get(opts, :target, @default_target)
    num_warps = positive_integer!(opts, :num_warps, @default_num_warps)
    num_stages = positive_integer!(opts, :num_stages, @default_num_stages)
    from_ttgir? = boolean!(opts, :from_ttgir, false)
    remove_layouts? = boolean!(opts, :remove_layout_conversions, true)
    capability = capability!(target)

    passes =
      case profile do
        :legacy ->
          legacy_passes(target, capability, num_warps, from_ttgir?, remove_layouts?)

        :pinned_nvidia_sm100 ->
          if capability < 100 do
            raise ArgumentError,
                  "unsupported_pipeline_profile: :pinned_nvidia_sm100 requires cuda:100 or newer"
          end

          pinned_sm100_passes(target, capability, num_warps, num_stages, from_ttgir?)

        other ->
          raise ArgumentError, "unsupported_pipeline_profile: #{inspect(other)}"
      end

    %__MODULE__{
      schema: @schema,
      profile: profile,
      target: target,
      capability: capability,
      num_warps: num_warps,
      num_stages: num_stages,
      provenance: if(profile == :pinned_nvidia_sm100, do: @pinned_provenance),
      passes: passes
    }
  end

  @doc "Returns a stable SHA-256 identity for a complete plan."
  @spec digest(t()) :: String.t()
  def digest(%__MODULE__{} = plan) do
    plan
    |> Map.from_struct()
    |> :erlang.term_to_binary([:deterministic])
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp legacy_passes(target, capability, num_warps, from_ttgir?, remove_layouts?) do
    [
      unless(from_ttgir?,
        do:
          pass(
            :convert_to_ttgpu,
            :ttgir,
            "convert-triton-to-tritongpu{target=#{target}, num-warps=#{num_warps}}"
          )
      ),
      pass(:coalesce, :ttgir, "tritongpu-coalesce"),
      pass(:f32_dot_tc, :ttgir, "tritongpu-F32DotTC"),
      pass(:plan_cta, :ttgir, "triton-nvidia-gpu-plan-cta"),
      if(remove_layouts?,
        do: pass(:remove_layout_conversions_pre, :ttgir, "tritongpu-remove-layout-conversions")
      ),
      pass(:optimize_thread_locality, :ttgir, "tritongpu-optimize-thread-locality"),
      pass(:accelerate_matmul, :ttgir, "tritongpu-accelerate-matmul"),
      if(remove_layouts?,
        do: pass(:remove_layout_conversions_post, :ttgir, "tritongpu-remove-layout-conversions")
      ),
      pass(:optimize_dot_operands, :ttgir, "tritongpu-optimize-dot-operands"),
      pass(:legacy_canonicalize, :ttgir, "canonicalize")
    ]
    |> Enum.reject(&is_nil/1)
    |> Kernel.++(llir_passes(capability))
  end

  defp pinned_sm100_passes(target, capability, num_warps, num_stages, from_ttgir?) do
    [
      unless(from_ttgir?,
        do:
          pass(
            :convert_to_ttgpu,
            :ttgir,
            "convert-triton-to-tritongpu{target=#{target} num-warps=#{num_warps} " <>
              "threads-per-warp=32 num-ctas=1}"
          )
      ),
      pass(:coalesce, :ttgir, "tritongpu-coalesce"),
      pass(:f32_dot_tc, :ttgir, "tritongpu-F32DotTC{emu-tf32=true}"),
      pass(:plan_cta, :ttgir, "triton-nvidia-gpu-plan-cta"),
      pass(:remove_layout_conversions_pre, :ttgir, "tritongpu-remove-layout-conversions"),
      pass(:optimize_thread_locality, :ttgir, "tritongpu-optimize-thread-locality"),
      pass(:accelerate_matmul, :ttgir, "tritongpu-accelerate-matmul"),
      pass(:remove_layout_conversions_post, :ttgir, "tritongpu-remove-layout-conversions"),
      pass(
        :optimize_dot_operands_pre,
        :ttgir,
        "tritongpu-optimize-dot-operands{hoist-layout-conversion=true}"
      ),
      pass(
        :optimize_descriptor_encoding,
        :ttgir,
        "triton-nvidia-optimize-descriptor-encoding"
      ),
      pass(:loop_aware_cse_pre, :ttgir, "triton-loop-aware-cse"),
      pass(:fuse_nested_loops, :ttgir, "tritongpu-fuse-nested-loops"),
      pass(:canonicalize_pre_pipeline, :ttgir, "canonicalize"),
      pass(:triton_licm, :ttgir, "triton-licm"),
      pass(:optimize_accumulator_init, :ttgir, "tritongpu-optimize-accumulator-init"),
      pass(:hoist_tmem_alloc_pre, :ttgir, "tritongpu-hoist-tmem-alloc{post-pipeline=false}"),
      pass(:promote_lhs_to_tmem, :ttgir, "tritongpu-promote-lhs-to-tmem"),
      pass(
        :assign_latencies,
        :ttgir,
        "tritongpu-assign-latencies{num-stages=#{num_stages}}"
      ),
      pass(:schedule_loops, :ttgir, "tritongpu-schedule-loops"),
      pass(
        :warp_specialize,
        :ttgir,
        "tritongpu-automatic-warp-specialization{num-stages=#{num_stages}}"
      ),
      pass(
        :pipeline,
        :ttgir,
        "tritongpu-pipeline{num-stages=#{num_stages} dump-intermediate-steps=false}"
      ),
      pass(:optimize_partition_warps, :ttgir, "tritongpu-optimize-partition-warps"),
      pass(:combine_tensor_select_and_if_ttgir, :ttgir, "tritongpu-combine-tensor-select-and-if"),
      pass(:hoist_tmem_alloc_post, :ttgir, "tritongpu-hoist-tmem-alloc{post-pipeline=true}"),
      pass(:remove_tmem_tokens, :ttgir, "triton-nvidia-gpu-remove-tmem-tokens"),
      pass(:canonicalize_post_pipeline, :ttgir, "canonicalize"),
      pass(:loop_aware_cse_post, :ttgir, "triton-loop-aware-cse"),
      pass(
        :optimize_dot_operands_post,
        :ttgir,
        "tritongpu-optimize-dot-operands{hoist-layout-conversion=true}"
      ),
      pass(:coalesce_async_copy, :ttgir, "tritongpu-coalesce-async-copy"),
      pass(:optimize_tmem_layouts, :ttgir, "triton-nvidia-optimize-tmem-layouts"),
      pass(:tmem_load_reduce, :ttgir, "triton-nvidia-tmem-load-reduce"),
      pass(:tma_lowering, :ttgir, "triton-nvidia-tma-lowering"),
      pass(:lower_clc, :ttgir, "triton-nvidia-gpu-lower-clc"),
      pass(:remove_layout_conversions_final, :ttgir, "tritongpu-remove-layout-conversions"),
      pass(:interleave_tmem, :ttgir, "triton-nvidia-interleave-tmem"),
      pass(:reduce_data_duplication, :ttgir, "tritongpu-reduce-data-duplication"),
      pass(:reorder_instructions, :ttgir, "tritongpu-reorder-instructions"),
      pass(:loop_aware_cse_final, :ttgir, "triton-loop-aware-cse"),
      pass(:symbol_dce, :ttgir, "symbol-dce"),
      pass(
        :fence_insertion,
        :ttgir,
        "triton-nvidia-gpu-fence-insertion{compute-capability=#{capability}}"
      ),
      pass(:lower_mma, :ttgir, "triton-nvidia-mma-lowering"),
      pass(:sccp, :ttgir, "sccp"),
      pass(:cse, :ttgir, "cse"),
      pass(:canonicalize_final, :ttgir, "canonicalize")
    ]
    |> Enum.reject(&is_nil/1)
    |> Kernel.++(llir_passes(capability))
  end

  defp llir_passes(capability) do
    [
      pass(:combine_tensor_select_and_if_llir, :llir, "tritongpu-combine-tensor-select-and-if"),
      pass(:allocate_warp_groups, :llir, "tritongpu-allocate-warp-groups"),
      pass(:convert_scf_to_cf, :llir, "convert-scf-to-cf"),
      pass(
        :allocate_shared_memory,
        :llir,
        "allocate-shared-memory-nv{compute-capability=#{capability} ptx-version=#{capability}}"
      ),
      pass(:tensor_memory_allocation, :llir, "triton-tensor-memory-allocation"),
      pass(:check_matmul_two_cta, :llir, "triton-nvidia-check-matmul-two-cta"),
      pass(:proxy_fence_insertion, :llir, "triton-nvidia-gpu-proxy-fence-insertion"),
      pass(:tmem_barrier_insertion, :llir, "triton-nvidia-gpu-tmem-barrier-insertion"),
      pass(
        :convert_ttgpu_to_llvm,
        :llir,
        "convert-triton-gpu-to-llvm{compute-capability=#{capability} ptx-version=#{capability}}"
      ),
      pass(:convert_warp_specialize_to_llvm, :llir, "convert-warp-specialize-to-llvm"),
      pass(:convert_nvgpu_to_llvm, :llir, "convert-nv-gpu-to-llvm"),
      pass(:convert_nvvm_to_llvm, :llir, "convert-nvvm-to-llvm"),
      pass(:llir_canonicalize, :llir, "canonicalize")
    ]
  end

  defp pass(id, phase, pipeline), do: %{id: id, phase: phase, pipeline: pipeline}

  defp capability!(target) when is_binary(target) do
    case Regex.run(~r/^cuda:(\d+)$/, target) do
      [_, capability] ->
        String.to_integer(capability)

      _ ->
        raise ArgumentError,
              "unsupported_pipeline_profile: invalid CUDA target #{inspect(target)}"
    end
  end

  defp capability!(target),
    do:
      raise(ArgumentError, "unsupported_pipeline_profile: invalid CUDA target #{inspect(target)}")

  defp positive_integer!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      value when is_integer(value) and value > 0 ->
        value

      value ->
        raise ArgumentError, "invalid_#{key}: expected a positive integer, got #{inspect(value)}"
    end
  end

  defp boolean!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      value when is_boolean(value) -> value
      value -> raise ArgumentError, "invalid_#{key}: expected a boolean, got #{inspect(value)}"
    end
  end
end
