defmodule Beaver.MLIR.Triton.PipelinePlanTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR.Triton.PipelinePlan

  @pinned_ids [
    :convert_to_ttgpu,
    :coalesce,
    :f32_dot_tc,
    :plan_cta,
    :remove_layout_conversions_pre,
    :optimize_thread_locality,
    :accelerate_matmul,
    :remove_layout_conversions_post,
    :optimize_dot_operands_pre,
    :optimize_descriptor_encoding,
    :loop_aware_cse_pre,
    :fuse_nested_loops,
    :canonicalize_pre_pipeline,
    :triton_licm,
    :optimize_accumulator_init,
    :hoist_tmem_alloc_pre,
    :promote_lhs_to_tmem,
    :assign_latencies,
    :schedule_loops,
    :warp_specialize,
    :pipeline,
    :optimize_partition_warps,
    :combine_tensor_select_and_if_ttgir,
    :hoist_tmem_alloc_post,
    :remove_tmem_tokens,
    :canonicalize_post_pipeline,
    :loop_aware_cse_post,
    :optimize_dot_operands_post,
    :coalesce_async_copy,
    :optimize_tmem_layouts,
    :tmem_load_reduce,
    :tma_lowering,
    :lower_clc,
    :remove_layout_conversions_final,
    :interleave_tmem,
    :reduce_data_duplication,
    :reorder_instructions,
    :loop_aware_cse_final,
    :symbol_dce,
    :fence_insertion,
    :lower_mma,
    :sccp,
    :cse,
    :canonicalize_final,
    :combine_tensor_select_and_if_llir,
    :allocate_warp_groups,
    :convert_scf_to_cf,
    :allocate_shared_memory,
    :tensor_memory_allocation,
    :check_matmul_two_cta,
    :proxy_fence_insertion,
    :tmem_barrier_insertion,
    :convert_ttgpu_to_llvm,
    :convert_warp_specialize_to_llvm,
    :convert_nvgpu_to_llvm,
    :convert_nvvm_to_llvm,
    :llir_canonicalize
  ]

  test "freezes the pinned sm_100+ order and provenance" do
    plan =
      PipelinePlan.build(
        target: "cuda:120",
        num_warps: 8,
        pipeline_profile: :pinned_nvidia_sm100
      )

    assert plan.schema == "beaver.triton.pipeline-plan.v1"
    assert plan.profile == :pinned_nvidia_sm100
    assert plan.capability == 120
    assert plan.num_stages == 3
    assert Enum.map(plan.passes, & &1.id) == @pinned_ids

    assert plan.provenance == %{
             triton_identity: "7d0817d08273d205",
             source_path: "third_party/nvidia/backend/compiler.py",
             source_symbol: "CUDABackend.make_ttgir",
             source_sha256: "d2b6c6b79146143fa56107133ddceec6369812d29e74cb362a7d400476765be8"
           }

    assert PipelinePlan.digest(plan) == PipelinePlan.digest(plan)
  end

  test "num_stages has exactly three owning pass entries" do
    for stages <- 1..3 do
      plan =
        PipelinePlan.build(
          target: "cuda:120",
          pipeline_profile: :pinned_nvidia_sm100,
          num_stages: stages
        )

      owners =
        Enum.filter(plan.passes, fn pass ->
          pass.id in [:assign_latencies, :warp_specialize, :pipeline]
        end)

      assert length(owners) == 3
      assert Enum.all?(owners, &String.contains?(&1.pipeline, "num-stages=#{stages}"))

      refute Enum.any?(plan.passes -- owners, fn pass ->
               String.contains?(pass.pipeline, "num-stages=")
             end)
    end
  end

  test "legacy is the default and retains the historical shortened TTGIR plan" do
    plan = PipelinePlan.build(target: "cuda:120", num_warps: 8)

    assert plan.profile == :legacy
    assert plan.provenance == nil

    assert Enum.map(Enum.filter(plan.passes, &(&1.phase == :ttgir)), & &1.pipeline) == [
             "convert-triton-to-tritongpu{target=cuda:120, num-warps=8}",
             "tritongpu-coalesce",
             "tritongpu-F32DotTC",
             "triton-nvidia-gpu-plan-cta",
             "tritongpu-remove-layout-conversions",
             "tritongpu-optimize-thread-locality",
             "tritongpu-accelerate-matmul",
             "tritongpu-remove-layout-conversions",
             "tritongpu-optimize-dot-operands",
             "canonicalize"
           ]
  end

  test "from_ttgir omits conversion from either profile" do
    for profile <- [:legacy, :pinned_nvidia_sm100] do
      plan = PipelinePlan.build(target: "cuda:120", pipeline_profile: profile, from_ttgir: true)
      refute Enum.any?(plan.passes, &(&1.id == :convert_to_ttgpu))
    end
  end

  test "rejects unsupported profiles, capabilities, and stage counts" do
    assert_raise ArgumentError, ~r/unsupported_pipeline_profile/, fn ->
      PipelinePlan.build(target: "cuda:120", pipeline_profile: :unknown)
    end

    assert_raise ArgumentError, ~r/requires cuda:100 or newer/, fn ->
      PipelinePlan.build(target: "cuda:90", pipeline_profile: :pinned_nvidia_sm100)
    end

    for invalid <- [0, -1, 1.5, "3"] do
      assert_raise ArgumentError, ~r/invalid_num_stages/, fn ->
        PipelinePlan.build(
          target: "cuda:120",
          pipeline_profile: :pinned_nvidia_sm100,
          num_stages: invalid
        )
      end
    end
  end

  test "locates the first pass or digest divergence" do
    common = %{index: 0, id: :coalesce, ir_sha256: "same"}
    left = [common, %{index: 1, id: :legacy_canonicalize, ir_sha256: "left"}]
    right = [common, %{index: 1, id: :f32_dot_tc, ir_sha256: "right"}]

    assert {:ok, %{index: 1, left: left_record, right: right_record}} =
             Beaver.Triton.first_trace_divergence(left, right)

    assert left_record == Enum.at(left, 1)
    assert right_record == Enum.at(right, 1)
    assert Beaver.Triton.first_trace_divergence(left, left) == :none
    assert Beaver.Triton.first_trace_divergence([], []) == :none
  end
end
