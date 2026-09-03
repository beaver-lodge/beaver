defmodule Beaver.MLIR.Triton.PipelineEvidence do
  @moduledoc """
  Content-addressed, zero-device evidence for a Triton pipeline differential.

  Raw MLIR, LLVM IR, and PTX are reduced to byte counts and canonical SHA-256
  identities.  Prefix traces contain the same digest-only records returned by
  `Beaver.Triton.compile_to_llvm_with_trace/2`.
  """

  alias Beaver.MLIR.Triton.PipelinePlan

  @schema "beaver.triton.pipeline-evidence.v1"
  @register_limit 128
  @shared_memory_limit 16_384

  @legacy_reference %{
    ttir: %{
      bytes: 10_063,
      digest: "52fa4ff7f135828ac91dec834dfca5dc319b96e3ae5315d416a16fd4c0111243"
    },
    ttgir: %{
      bytes: 14_495,
      digest: "7fe8cf171e750c8d26661aa6e101ef871563a956f66a7abbbcc7a604651a0f39"
    },
    lowered_mlir: %{
      bytes: 219_691,
      digest: "6ff831c7c843ae8dc7b48f187905ec7d167e36ae8837404a125d63006357aa32"
    },
    llvm_ir: %{
      bytes: 160_807,
      digest: "561afbfe7b2f9b84e2571a8626ea76aa30e72e31f0c10728c931f9536aa8ebe1"
    },
    ptx: %{
      bytes: 36_182,
      digest: "562cccb7551773ed42c97c47c2aba07828449f9056c9235800ed04a5d84c2d38"
    },
    shared_memory_bytes: 32_768,
    registers_per_thread: 170,
    stack_frame_bytes: 0,
    spill_store_bytes: 0,
    spill_load_bytes: 0
  }

  @mismatch_domains [
    :plan_golden_mismatch,
    :plan_provenance_missing,
    :unsupported_pipeline_profile,
    :invalid_num_stages,
    :num_stages_injection_missing,
    :pass_registration_missing,
    :profile_compile_failed,
    :prefix_verification_failed,
    :legacy_digest_drift,
    :smoke_dot_spurious,
    :native_tensor_missing,
    :resource_viable,
    :resource_unchanged,
    :resource_regressed,
    :artifact_incomplete,
    :classification_inconclusive,
    :seal_invalid
  ]

  @doc "Returns the immutable #291 legacy artifact reference."
  @spec legacy_reference() :: map()
  def legacy_reference, do: @legacy_reference

  @doc "Returns the closed mismatch and classification vocabulary."
  @spec mismatch_domains() :: [atom()]
  def mismatch_domains, do: @mismatch_domains

  @doc "Builds a digest-only pipeline evidence seal from compiler observations."
  @spec build(map()) :: map() | {:error, :artifact_incomplete}
  def build(%{
        identity: identity,
        fixture: fixture,
        plan: %PipelinePlan{} = plan,
        legacy: legacy,
        candidate: candidate,
        attempt_ledger: attempt_ledger
      }) do
    with true <- complete_identity?(identity),
         {:ok, fixture_record} <- fixture_record(fixture),
         {:ok, legacy_record} <- artifact_record(legacy),
         {:ok, candidate_record} <- artifact_record(candidate),
         true <- complete_attempt_ledger?(attempt_ledger) do
      legacy_match? = legacy_match?(fixture_record, legacy_record)
      pipeline_classification = pipeline_classification(plan, legacy_match?, candidate_record)
      resource_classification = resource_classification(legacy_record, candidate_record)

      core = %{
        schema: @schema,
        issue: 196,
        authority: :zero_device_compiler_and_ptxas,
        identity: identity,
        fixture: fixture_record,
        plan: %{
          profile: plan.profile,
          digest: PipelinePlan.digest(plan),
          provenance: plan.provenance,
          num_stages: plan.num_stages,
          pass_count: length(plan.passes)
        },
        legacy_reference: @legacy_reference,
        legacy: legacy_record,
        candidate: candidate_record,
        pipeline_classification: pipeline_classification,
        resource_classification: resource_classification,
        downstream_authorized:
          pipeline_classification == :PARITY_PROFILE_GO and
            resource_classification == :RESOURCE_VIABLE,
        attempt_ledger: attempt_ledger,
        mismatch_domains: @mismatch_domains
      }

      Map.put(core, :seal_digest, digest(core))
    else
      _ -> {:error, :artifact_incomplete}
    end
  end

  def build(_input), do: {:error, :artifact_incomplete}

  @doc "Verifies the content-addressed seal without trusting its classifications."
  @spec verify(map()) :: :ok | {:error, :seal_invalid | :artifact_incomplete}
  def verify(%{seal_digest: seal_digest} = evidence) when is_binary(seal_digest) do
    if digest(Map.delete(evidence, :seal_digest)) == seal_digest,
      do: :ok,
      else: {:error, :seal_invalid}
  end

  def verify(_evidence), do: {:error, :artifact_incomplete}

  @doc "Parses the stable resource fields emitted by `ptxas -v`."
  @spec parse_ptxas(String.t()) :: {:ok, map()} | {:error, :artifact_incomplete}
  def parse_ptxas(output) when is_binary(output) do
    with {:ok, registers} <- capture_integer(output, ~r/Used (\d+) registers/),
         {:ok, stack} <- capture_integer(output, ~r/(\d+) bytes stack frame/),
         {:ok, spill_stores} <- capture_integer(output, ~r/(\d+) bytes spill stores/),
         {:ok, spill_loads} <- capture_integer(output, ~r/(\d+) bytes spill loads/) do
      {:ok,
       %{
         registers_per_thread: registers,
         stack_frame_bytes: stack,
         spill_store_bytes: spill_stores,
         spill_load_bytes: spill_loads
       }}
    else
      _ -> {:error, :artifact_incomplete}
    end
  end

  def parse_ptxas(_output), do: {:error, :artifact_incomplete}

  @doc "Canonical digest compatible with the Accendium #291 evidence format."
  @spec digest(term()) :: String.t()
  def digest(value) do
    value
    |> encode()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp fixture_record(%{ttir: ttir, ttgir: ttgir})
       when is_binary(ttir) and is_binary(ttgir) do
    {:ok, %{ttir: artifact(ttir), ttgir: artifact(ttgir)}}
  end

  defp fixture_record(_fixture), do: {:error, :artifact_incomplete}

  defp artifact_record(%{
         lowered_mlir: lowered_mlir,
         llvm_ir: llvm_ir,
         ptx: ptx,
         shared_memory_bytes: shared_memory_bytes,
         resources: resources,
         trace: trace
       })
       when is_binary(lowered_mlir) and is_binary(llvm_ir) and is_binary(ptx) and
              is_integer(shared_memory_bytes) and is_map(resources) and is_list(trace) do
    required_resources = [
      :registers_per_thread,
      :stack_frame_bytes,
      :spill_store_bytes,
      :spill_load_bytes
    ]

    if Enum.all?(required_resources, &is_integer(resources[&1])) and complete_trace?(trace) do
      {:ok,
       %{
         lowered_mlir: artifact(lowered_mlir),
         llvm_ir: artifact(llvm_ir),
         ptx: artifact(ptx),
         shared_memory_bytes: shared_memory_bytes,
         resources: Map.take(resources, required_resources),
         native_tensor: String.contains?(ptx, "mma.sync"),
         trace: trace,
         trace_digest: digest(trace)
       }}
    else
      {:error, :artifact_incomplete}
    end
  end

  defp artifact_record(_artifact), do: {:error, :artifact_incomplete}

  defp artifact(content), do: %{bytes: byte_size(content), digest: digest(content)}

  defp legacy_match?(fixture, legacy) do
    fixture.ttir == @legacy_reference.ttir and
      fixture.ttgir == @legacy_reference.ttgir and
      legacy.lowered_mlir == @legacy_reference.lowered_mlir and
      legacy.llvm_ir == @legacy_reference.llvm_ir and
      legacy.ptx == @legacy_reference.ptx and
      legacy.shared_memory_bytes == @legacy_reference.shared_memory_bytes and
      Enum.all?(legacy.resources, fn {key, value} -> @legacy_reference[key] == value end)
  end

  defp pipeline_classification(plan, true, %{native_tensor: true, trace: trace}) do
    if plan.profile == :pinned_nvidia_sm100 and
         plan.provenance == PipelinePlan.pinned_provenance() and
         trace_matches_plan?(trace, plan),
       do: :PARITY_PROFILE_GO,
       else: :PARITY_PROFILE_FAILED
  end

  defp pipeline_classification(_plan, _legacy_match?, _candidate),
    do: :PARITY_PROFILE_FAILED

  defp resource_classification(legacy, candidate) do
    cond do
      resource_viable?(candidate) ->
        :RESOURCE_VIABLE

      resource_regressed?(legacy, candidate) ->
        :RESOURCE_REGRESSED

      true ->
        :RESOURCE_UNCHANGED
    end
  end

  defp resource_viable?(candidate) do
    candidate.resources.registers_per_thread <= @register_limit and
      candidate.shared_memory_bytes <= @shared_memory_limit and
      zero_stack_and_spill?(candidate.resources)
  end

  defp resource_regressed?(legacy, candidate) do
    candidate.resources.registers_per_thread > legacy.resources.registers_per_thread or
      candidate.shared_memory_bytes > legacy.shared_memory_bytes or
      candidate.resources.stack_frame_bytes > legacy.resources.stack_frame_bytes or
      candidate.resources.spill_store_bytes > legacy.resources.spill_store_bytes or
      candidate.resources.spill_load_bytes > legacy.resources.spill_load_bytes
  end

  defp zero_stack_and_spill?(resources) do
    resources.stack_frame_bytes == 0 and resources.spill_store_bytes == 0 and
      resources.spill_load_bytes == 0
  end

  defp complete_trace?(trace) do
    trace
    |> Enum.with_index()
    |> Enum.all?(fn
      {%{index: index, id: id, phase: phase, pipeline: pipeline, ir_sha256: digest}, index}
      when is_atom(id) and phase in [:ttgir, :llir] and is_binary(pipeline) and
             byte_size(digest) == 64 ->
        true

      _ ->
        false
    end)
  end

  defp trace_matches_plan?(trace, plan) do
    plan_digest = PipelinePlan.digest(plan)

    Enum.zip(trace, plan.passes)
    |> Enum.all?(fn {record, pass} ->
      record.id == pass.id and record.phase == pass.phase and record.pipeline == pass.pipeline and
        record.plan_digest == plan_digest
    end) and length(trace) == length(plan.passes)
  end

  defp complete_identity?(identity) when is_map(identity) do
    Enum.all?([:beaver, :kinda, :triton, :llvm, :cuda_target, :ptxas], fn key ->
      is_binary(identity[key]) and identity[key] != ""
    end)
  end

  defp complete_identity?(_identity), do: false

  defp complete_attempt_ledger?(%{cpu_compiles: cpu, gpu: 0, ncu: 0, nsys: 0, retries: retries}),
    do: is_integer(cpu) and cpu > 0 and is_integer(retries) and retries >= 0

  defp complete_attempt_ledger?(_ledger), do: false

  defp capture_integer(output, regex) do
    case Regex.run(regex, output) do
      [_, value] -> {:ok, String.to_integer(value)}
      _ -> {:error, :artifact_incomplete}
    end
  end

  defp encode(nil), do: "n"
  defp encode(true), do: "b1"
  defp encode(false), do: "b0"
  defp encode(value) when is_integer(value), do: ["i", Integer.to_string(value), ";"]
  defp encode(value) when is_float(value), do: ["f", :erlang.float_to_binary(value), ";"]
  defp encode(value) when is_atom(value), do: encode(Atom.to_string(value))

  defp encode(value) when is_binary(value),
    do: ["s", Integer.to_string(byte_size(value)), ":", value]

  defp encode(value) when is_list(value),
    do: ["l", Integer.to_string(length(value)), ":", Enum.map(value, &encode/1)]

  defp encode(value) when is_tuple(value),
    do: [
      "t",
      Integer.to_string(tuple_size(value)),
      ":",
      value |> Tuple.to_list() |> Enum.map(&encode/1)
    ]

  defp encode(%_struct{} = value), do: value |> Map.from_struct() |> encode()

  defp encode(value) when is_map(value) do
    pairs = value |> Enum.map(fn {key, item} -> {to_string(key), item} end) |> Enum.sort()
    ["m", Integer.to_string(length(pairs)), ":", Enum.map(pairs, &encode_pair/1)]
  end

  defp encode_pair({key, value}), do: [encode(key), encode(value)]
end
