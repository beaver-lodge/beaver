defmodule Beaver.MLIR.Triton.PipelineEvidenceTest do
  use Beaver.Case, async: false

  alias Beaver.MLIR
  alias Beaver.MLIR.Triton.{PipelineEvidence, PipelinePlan}

  @moduletag :triton
  @enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil
  @ptxas System.find_executable("ptxas")
  @fixture "test/fixtures/triton/ttir_sm120_q16_kv64_k256.mlir"

  setup_all do
    if @enabled, do: Beaver.Triton.register_passes()
    :ok
  end

  test "parses complete ptxas resource output and rejects partial output" do
    output = """
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info : Used 80 registers, used 1 barriers
    """

    assert {:ok,
            %{
              registers_per_thread: 80,
              stack_frame_bytes: 0,
              spill_store_bytes: 0,
              spill_load_bytes: 0
            }} = PipelineEvidence.parse_ptxas(output)

    assert {:error, :artifact_incomplete} = PipelineEvidence.parse_ptxas("Used 80 registers")
  end

  @tag skip: !(@enabled and @ptxas)
  test "seals the #291 legacy reproduction and pinned-profile resource classification" do
    source = File.read!(@fixture)
    context = MLIR.Context.create(all_dialects: false)
    on_exit(fn -> MLIR.Context.destroy(context) end)
    Beaver.Triton.register(context)

    ttgir = lower_to_ttgir(source, context)
    legacy = observation(ttgir, context, :legacy)
    candidate = observation(ttgir, context, :pinned_nvidia_sm100)

    plan =
      PipelinePlan.build(
        target: "cuda:120",
        num_warps: 8,
        from_ttgir: true,
        pipeline_profile: :pinned_nvidia_sm100
      )

    input = %{
      identity: %{
        beaver: "dbb07f6a328b4c73df7c756d8f64b4b48af6b6f2",
        kinda: "cb5a5021b1f7538ace89936f0449b4ef827c5a71",
        triton: "7d0817d08273d205",
        llvm: "24.0.0git",
        cuda_target: "cuda:120",
        ptxas: "13.1"
      },
      fixture: %{ttir: source, ttgir: ttgir},
      plan: plan,
      legacy: legacy,
      candidate: candidate,
      attempt_ledger: %{cpu_compiles: 5, gpu: 0, ncu: 0, nsys: 0, retries: 0}
    }

    evidence = PipelineEvidence.build(input)

    assert evidence.legacy_reference == PipelineEvidence.legacy_reference()
    assert evidence.legacy.lowered_mlir == evidence.legacy_reference.lowered_mlir
    assert evidence.legacy.llvm_ir == evidence.legacy_reference.llvm_ir
    assert evidence.legacy.ptx == evidence.legacy_reference.ptx
    assert evidence.legacy.resources.registers_per_thread == 170
    assert evidence.legacy.shared_memory_bytes == 32_768

    assert evidence.pipeline_classification == :PARITY_PROFILE_GO
    assert evidence.resource_classification == :RESOURCE_REGRESSED
    refute evidence.downstream_authorized
    assert evidence.candidate.native_tensor
    assert evidence.candidate.resources.registers_per_thread == 80
    assert evidence.candidate.shared_memory_bytes == 141_312
    assert evidence.candidate.resources.stack_frame_bytes == 0
    assert evidence.candidate.resources.spill_store_bytes == 0
    assert evidence.candidate.resources.spill_load_bytes == 0
    assert length(evidence.candidate.trace) == 56
    assert evidence.attempt_ledger == %{cpu_compiles: 5, gpu: 0, ncu: 0, nsys: 0, retries: 0}
    assert :ok = PipelineEvidence.verify(evidence)

    assert {:error, :seal_invalid} =
             evidence
             |> put_in([:candidate, :shared_memory_bytes], 16_384)
             |> PipelineEvidence.verify()

    drifted_trace =
      update_in(candidate.trace, [Access.at(0), :plan_digest], fn _digest ->
        String.duplicate("0", 64)
      end)

    drifted = PipelineEvidence.build(put_in(input, [:candidate, :trace], drifted_trace))
    assert drifted.pipeline_classification == :PARITY_PROFILE_FAILED
    refute drifted.downstream_authorized
  end

  defp lower_to_ttgir(source, context) do
    source
    |> MLIR.Module.create!(ctx: context)
    |> Beaver.Composer.append("convert-triton-to-tritongpu{target=cuda:120 num-warps=8}")
    |> Beaver.Composer.run!()
    |> MLIR.to_string()
  end

  defp observation(ttgir, context, profile) do
    module = MLIR.Module.create!(ttgir, ctx: context)

    lowered =
      Beaver.Triton.compile_to_llvm(module,
        target: "cuda:120",
        num_warps: 8,
        from_ttgir: true,
        pipeline_profile: profile
      )

    lowered_mlir = MLIR.to_string(lowered)
    llvm_ir = Beaver.MLIR.Target.LLVMIR.translate!(lowered)
    ptx = Beaver.MLIR.Target.LLVMIR.compile_to_ptx!(llvm_ir, cpu: "sm_120")

    trace_module = MLIR.Module.create!(ttgir, ctx: context)

    {_trace_lowered, trace} =
      Beaver.Triton.compile_to_llvm_with_trace(trace_module,
        target: "cuda:120",
        num_warps: 8,
        from_ttgir: true,
        pipeline_profile: profile
      )

    resources = ptxas_resources!(ptx, profile)

    %{
      lowered_mlir: lowered_mlir,
      llvm_ir: llvm_ir,
      ptx: ptx,
      shared_memory_bytes: shared_memory(lowered_mlir),
      resources: resources,
      trace: trace
    }
  end

  defp ptxas_resources!(ptx, profile) do
    path =
      Path.join(
        System.tmp_dir!(),
        "beaver-196-#{profile}-#{System.unique_integer([:positive])}.ptx"
      )

    output = Path.rootname(path) <> ".cubin"
    File.write!(path, ptx)

    try do
      {diagnostics, 0} =
        System.cmd(@ptxas, ["-arch=sm_120", "-v", "-o", output, path], stderr_to_stdout: true)

      {:ok, resources} = PipelineEvidence.parse_ptxas(diagnostics)
      resources
    after
      File.rm(path)
      File.rm(output)
    end
  end

  defp shared_memory(lowered_mlir) do
    [_, bytes] = Regex.run(~r/ttg\.shared = (\d+)/, lowered_mlir)
    String.to_integer(bytes)
  end
end
