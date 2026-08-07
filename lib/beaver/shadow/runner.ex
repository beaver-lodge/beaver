defmodule Beaver.Shadow.Runner do
  @moduledoc """
  A CPU-only, replayable compilation experiment loop.

  Given an MLIR input and a Transform schedule, the runner:

  1. enumerates deterministic candidates;
  2. resolves each candidate and evaluates it through an injectable evaluator;
  3. records one `Beaver.Shadow.Receipt` per candidate;
  4. returns the receipts and a winner that can be replayed from serialized
     bytecode without re-running the resolver.

  The evaluator owns scoring. Beaver only records values, metadata, failures,
  and cache/action evidence in the receipt. Durations are observations and
  never participate in `Receipt.identity/1`.
  """

  alias Beaver.MLIR
  alias MLIR.Transform
  alias MLIR.Transform.Schedule
  alias Beaver.Shadow.Receipt

  defmodule Run do
    @moduledoc "One experiment run: ordered candidate receipts plus the winner."
    @enforce_keys [:input_digest, :receipts, :winner]
    defstruct [:input_digest, :receipts, :winner]

    @type t() :: %__MODULE__{
            input_digest: String.t(),
            receipts: [Receipt.t()],
            winner: Receipt.t() | nil
          }
  end

  @type evaluator() ::
          (Schedule.Resolved.t(), %{index: non_neg_integer(), choices: map()} ->
             {:ok, term(), term()} | {:ok, term()} | {:error, term()})

  @doc """
  Runs one experiment over all candidates of `schedule` against `input`.

  Options:

    * `:sequence` — the named sequence to analyze (`"__transform_main"` default)
    * `:evaluator` — `(Resolved, candidate) -> {:ok, value, metadata} | {:error, reason}`;
      defaults to a deterministic surrogate that scores by schedule digest and
      choices so the loop is runnable without a GPU or compilation cache
    * `:max_candidates` — enumeration bound
  """
  @spec run(MLIR.Module.t() | binary(), Schedule.input(), keyword()) :: Run.t()
  def run(input, schedule, opts \\ []) do
    sequence = Keyword.get(opts, :sequence, Schedule.sequence(schedule))
    evaluator = Keyword.get(opts, :evaluator, &default_evaluator/2)

    {:ok, candidates} =
      Schedule.enumerate(schedule,
        sequence: sequence,
        max_candidates: Keyword.get(opts, :max_candidates, 10_000)
      )

    input_digest = input |> source_bytes() |> digest()

    receipts =
      candidates
      |> Enum.with_index()
      |> Enum.map(fn {choices, index} ->
        candidate = %{index: index, choices: choices}
        resolved = Schedule.resolve!(schedule, choices, sequence: sequence)
        evaluate_and_record(input, resolved, candidate, evaluator)
      end)

    winner =
      receipts
      |> Enum.find(&(&1.status == :ok))

    %Run{input_digest: input_digest, receipts: receipts, winner: winner}
  end

  @doc "Replays a receipt's winner bytecode against a payload module."
  @spec replay(Receipt.t(), MLIR.Module.t(), keyword()) ::
          {:ok, MLIR.Transform.Result.t()} | {:error, Transform.Error.t()}
  def replay(%Receipt{schedule: %{bytecode: bytecode}} = receipt, payload, opts \\ [])
      when is_binary(bytecode) do
    MLIR.Transform.execute(payload, bytecode,
      sequence: receipt.schedule.sequence,
      expensive_checks: Keyword.get(opts, :expensive_checks, false)
    )
  end

  defp evaluate_and_record(input, resolved, candidate, evaluator) do
    started = System.monotonic_time()

    try do
      case evaluator.(resolved, candidate) do
        {:ok, value, metadata} ->
          record(input, resolved, candidate, :ok, value, metadata, started, nil)

        {:ok, value} ->
          record(input, resolved, candidate, :ok, value, %{}, started, nil)

        {:error, reason} ->
          record(input, resolved, candidate, :failed, nil, %{}, started, %{
            kind: :evaluation_failure,
            reason: reason
          })
      end
    rescue
      exception ->
        record(input, resolved, candidate, :failed, nil, %{}, started, %{
          kind: :evaluation_failure,
          reason: Exception.format(:error, exception, __STACKTRACE__)
        })
    end
  end

  defp record(input, resolved, candidate, status, value, user_metadata, started, failure) do
    duration = System.monotonic_time() - started

    schedule_record = %{
      sequence: resolved.sequence,
      digest: resolved.digest,
      text: resolved.text,
      bytecode: resolved.bytecode
    }

    artifact =
      Map.get(user_metadata, :artifact, %{cache: nil, lookup_key: nil, artifact_key: nil})

    trace =
      Map.get(user_metadata, :trace, %{
        action_count: 0,
        tags: [],
        candidate_index: candidate.index,
        schedule_digest: resolved.digest
      })

    receipt = %Receipt{
      format: Receipt.format(),
      source_digest: digest(source_bytes(input)),
      source_structural_hash: Map.get(user_metadata, :structural_hash),
      llvm_revision:
        Map.get(user_metadata, :llvm_revision, MLIR.CompilationRuntime.llvm_revision()),
      target: Map.get(user_metadata, :target),
      schema_version: Map.get(user_metadata, :schema_version),
      schedule: schedule_record,
      candidate: candidate,
      artifact: artifact,
      trace: trace,
      status: status,
      failure: failure,
      user_metadata: Map.put(user_metadata, :value, value) |> Map.put(:duration, duration)
    }

    receipt
  end

  @doc "Deterministic surrogate evaluator: score by digest and choices only."
  def default_evaluator(resolved, candidate) do
    {:ok, {resolved.digest, candidate.choices},
     %{
       artifact: %{cache: :miss, lookup_key: resolved.digest, artifact_key: resolved.digest},
       trace: %{
         action_count: 0,
         tags: [],
         candidate_index: candidate.index,
         schedule_digest: resolved.digest
       }
     }}
  end

  defp source_bytes(%MLIR.Module{} = module), do: MLIR.Bytecode.write!(module)
  defp source_bytes(binary) when is_binary(binary), do: binary

  defp digest(bytes), do: :crypto.hash(:sha256, bytes) |> Base.encode16(case: :lower)
end
