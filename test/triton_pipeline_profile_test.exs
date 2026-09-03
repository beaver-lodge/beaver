defmodule Beaver.MLIR.Triton.PipelineProfileTest do
  use Beaver.Case, async: false

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

  setup_all do
    if @enabled, do: Beaver.Triton.register_passes()
    :ok
  end

  @tag skip: !@enabled
  test "executes and deterministically traces the pinned sm_100+ profile" do
    {first_module, first_trace} = compile_with_trace(@load_store)
    first_text = MLIR.to_string(first_module, generic: true)

    assert first_text =~ "llvm.func"
    refute first_text =~ "tt.dot"
    refute first_text =~ "mma.sync"

    {second_module, second_trace} = compile_with_trace(@load_store)

    assert Enum.map(first_trace, &Map.drop(&1, [:plan_digest])) ==
             Enum.map(second_trace, &Map.drop(&1, [:plan_digest]))

    assert length(first_trace) == 57
    assert Enum.all?(first_trace, &(byte_size(&1.ir_sha256) == 64))
    assert Enum.map(first_trace, & &1.index) == Enum.to_list(0..56)
    assert hd(first_trace).id == :convert_to_ttgpu
    assert List.last(first_trace).id == :llir_canonicalize

    MLIR.Module.destroy(first_module)
    MLIR.Module.destroy(second_module)
  end

  @tag skip: !@enabled
  test "keeps legacy as the default and exposes its first profile divergence" do
    {default_module, default_trace} = compile_with_trace(@load_store, [])
    {legacy_module, legacy_trace} = compile_with_trace(@load_store, pipeline_profile: :legacy)

    assert MLIR.to_string(default_module, generic: true) ==
             MLIR.to_string(legacy_module, generic: true)

    assert default_trace == legacy_trace

    {pinned_module, pinned_trace} = compile_with_trace(@load_store)

    assert {:ok, %{index: index, left: left, right: right}} =
             Beaver.Triton.first_trace_divergence(legacy_trace, pinned_trace)

    assert index >= 0
    assert left != right

    for module <- [default_module, legacy_module, pinned_module], do: MLIR.Module.destroy(module)
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
