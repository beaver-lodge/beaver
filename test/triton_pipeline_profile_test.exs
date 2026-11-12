defmodule Beaver.MLIR.Triton.PipelineProfileTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  @moduletag :triton
  @enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  @load_store ~S"""
  module {
    tt.func public @pipeline_profile_smoke(
      %input: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %output: !tt.ptr<f32> {tt.divisibility = 16 : i32}
    ) attributes {noinline = false} {
      %value = tt.load %input : !tt.ptr<f32>
      tt.store %output, %value : !tt.ptr<f32>
      tt.return
    }
  }
  """

  # The producer spans four warps, while the consumer reads those values in
  # one warp. Allocation alone cannot make the shared writes visible to it.
  @shared_load ~S"""
  #producer = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
  #consumer = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
  #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
  module attributes {ttg.target = "__TARGET__", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @shared_load(%input: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %output: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
      %in_indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #producer>
      %in_base = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #producer>
      %in_ptrs = tt.addptr %in_base, %in_indices : tensor<128x!tt.ptr<f32>, #producer>, tensor<128xi32, #producer>
      %values = tt.load %in_ptrs : tensor<128x!tt.ptr<f32>, #producer>
      %buffer = ttg.local_alloc %values : (tensor<128xf32, #producer>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory>
      %result = ttg.local_load %buffer : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory> -> tensor<128xf32, #consumer>
      %out_indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #consumer>
      %out_base = tt.splat %output : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #consumer>
      %out_ptrs = tt.addptr %out_base, %out_indices : tensor<128x!tt.ptr<f32>, #consumer>, tensor<128xi32, #consumer>
      tt.store %out_ptrs, %result : tensor<128x!tt.ptr<f32>, #consumer>
      tt.return
    }
  }
  """

  setup_all do
    if @enabled, do: Beaver.Triton.register_passes()
    :ok
  end

  @tag skip: !@enabled
  test "legacy conversion keeps the exact target and warp attributes", %{ctx: ctx} do
    Beaver.Triton.register(ctx)
    plan = Beaver.Triton.pipeline_plan(target: "cuda:120", num_warps: 8)
    module = MLIR.Module.create!(@load_store, ctx: ctx)

    MLIR.Module.with_module(module, fn _ ->
      module = module |> Beaver.Composer.append(hd(plan.passes).pipeline) |> Beaver.Composer.run!()

      operation = MLIR.Operation.from_module(module)
      assert {:ok, target} = MLIR.Operation.fetch(operation, "ttg.target")
      assert MLIR.Attribute.value(target) == "cuda:120"
      assert {:ok, warps} = MLIR.Operation.fetch(operation, "ttg.num-warps")
      assert MLIR.Attribute.value(warps) == 8
    end)
  end

  @tag skip: !@enabled
  test "executes and deterministically traces the pinned sm_100+ profile" do
    {first_module, first_trace} = compile_with_trace(@load_store)
    first_text = MLIR.to_string(first_module, generic: true)
    assert first_text =~ "llvm.func"
    refute first_text =~ "tt.dot"
    refute first_text =~ "mma.sync"
    {second_module, second_trace} = compile_with_trace(@load_store)

    assert Enum.map(first_trace, &Map.drop(&1, [:plan_digest])) == Enum.map(second_trace, &Map.drop(&1, [:plan_digest]))

    assert length(first_trace) == 58
    assert Enum.all?(first_trace, &(byte_size(&1.ir_sha256) == 64))
    assert Enum.map(first_trace, & &1.index) == Enum.to_list(0..57)
    assert hd(first_trace).id == :convert_to_ttgpu
    assert List.last(first_trace).id == :llir_canonicalize
    MLIR.Module.destroy(first_module)
    MLIR.Module.destroy(second_module)
  end

  @tag skip: !@enabled
  test "keeps legacy as the default and exposes its first profile divergence" do
    {default_module, default_trace} = compile_with_trace(@load_store, [])
    {legacy_module, legacy_trace} = compile_with_trace(@load_store, pipeline_profile: :legacy)

    assert MLIR.to_string(default_module, generic: true) == MLIR.to_string(legacy_module, generic: true)

    assert default_trace == legacy_trace
    {pinned_module, pinned_trace} = compile_with_trace(@load_store)

    assert {:ok, %{index: index, left: left, right: right}} =
             Beaver.Triton.first_trace_divergence(legacy_trace, pinned_trace)

    assert index >= 0
    assert left != right
    for module <- [default_module, legacy_module, pinned_module], do: MLIR.Module.destroy(module)
  end

  @tag skip: !@enabled
  test "inserts cross-warp shared-memory synchronization before lowering", %{ctx: ctx} do
    Beaver.Triton.register(ctx)

    for capability <- [80, 120] do
      plan = Beaver.Triton.pipeline_plan(target: "cuda:#{capability}", from_ttgir: true)

      {allocation, [membar | lowering]} =
        plan.passes |> Enum.filter(&(&1.phase == :llir)) |> Enum.split_while(&(&1.id != :shared_memory_barriers))

      source = String.replace(@shared_load, "__TARGET__", "cuda:#{capability}")
      module = MLIR.Module.create!(source, ctx: ctx)

      MLIR.Module.with_module(module, fn _ ->
        run_passes(module, allocation)
        refute MLIR.to_string(module) =~ "ttg.barrier"
        run_passes(module, [membar])
        assert MLIR.to_string(module) =~ "ttg.barrier local"
        run_passes(module, lowering)

        ptx =
          module
          |> MLIR.Target.LLVMIR.translate!()
          |> MLIR.Target.LLVMIR.compile_to_ptx!(cpu: "sm_#{capability}")

        assert ptx =~ "bar.sync"
      end)
    end
  end

  defp run_passes(module, passes) do
    passes |> Enum.reduce(module, &Beaver.Composer.append(&2, &1.pipeline)) |> Beaver.Composer.run!() |> MLIR.verify!()
  end

  defp compile_with_trace(source, opts \\ :pinned) do
    context = MLIR.Context.create(all_dialects: false)
    Beaver.Triton.register(context)
    module = MLIR.Module.create!(source, ctx: context)

    options =
      case opts do
        :pinned -> [target: "cuda:120", pipeline_profile: :pinned_nvidia_sm100]
        explicit -> Keyword.put_new(explicit, :target, "cuda:120")
      end

    Beaver.Triton.compile_to_llvm_with_trace(module, options)
  end
end
