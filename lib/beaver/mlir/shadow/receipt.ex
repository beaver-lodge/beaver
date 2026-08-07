defmodule Beaver.MLIR.Shadow.Receipt do
  @moduledoc """
  A versioned, serializable record of one compilation experiment.

  A receipt closes the loop between an input, a Transform schedule candidate,
  the compilation artifact, and the action-tracing evidence collected for that
  candidate. It is deliberately a data record, not a runner: the same receipt
  can be produced by a Mix task, a Livebook, or an application-specific
  evaluator.

  ## Identity and replay

  `identity/1` hashes only stable facts (input digest, revisions, target,
  schedule identity, choices, and failure status). Durations and other
  observations are recorded in the receipt but never enter the identity, so
  identical experiments compare equal regardless of when they ran.

  The winner's resolved schedule bytecode is retained as `winner_bytecode` and
  can be replayed with `Beaver.MLIR.Transform.apply_named_sequence/3` without
  calling the resolver again.
  """

  @format 1

  @enforce_keys [:format, :source_digest, :schedule, :candidate, :artifact, :trace]
  defstruct [
    :format,
    :source_digest,
    :source_structural_hash,
    :llvm_revision,
    :target,
    :schema_version,
    :schedule,
    :candidate,
    :artifact,
    :trace,
    :status,
    :failure,
    :user_metadata
  ]

  @type schedule_record() :: %{
          sequence: String.t(),
          digest: String.t(),
          text: String.t() | nil,
          bytecode: binary() | nil
        }

  @type candidate_record() :: %{
          index: non_neg_integer(),
          choices: map()
        }

  @type artifact_record() :: %{
          cache: :hit | :miss,
          lookup_key: String.t() | nil,
          artifact_key: String.t() | nil
        }

  @type trace_record() :: %{
          action_count: non_neg_integer(),
          tags: [String.t()],
          candidate_index: non_neg_integer() | nil,
          schedule_digest: String.t() | nil
        }

  @type failure_record() :: %{
          kind: atom(),
          reason: term()
        }

  @type t() :: %__MODULE__{
          format: pos_integer(),
          source_digest: String.t(),
          source_structural_hash: non_neg_integer() | nil,
          llvm_revision: String.t() | nil,
          target: term(),
          schema_version: term(),
          schedule: schedule_record(),
          candidate: candidate_record(),
          artifact: artifact_record(),
          trace: trace_record(),
          status: :ok | :evaluated | :failed | nil,
          failure: failure_record() | nil,
          user_metadata: term()
        }

  @doc "Returns the current receipt format version."
  def format, do: @format

  @doc """
  Computes the stable identity of a receipt.

  Only provenance and selection facts participate; measurements such as
  durations are excluded by construction.
  """
  @spec identity(t()) :: String.t()
  def identity(%__MODULE__{} = receipt) do
    receipt
    |> stable_facts()
    |> then(&:crypto.hash(:sha256, :erlang.term_to_binary(&1)))
    |> Base.encode16(case: :lower)
  end

  @doc "The stable, identity-bearing fields of a receipt."
  @spec stable_facts(t()) :: map()
  def stable_facts(%__MODULE__{} = receipt) do
    %{
      format: receipt.format,
      source_digest: receipt.source_digest,
      source_structural_hash: receipt.source_structural_hash,
      llvm_revision: receipt.llvm_revision,
      target: receipt.target,
      schema_version: receipt.schema_version,
      schedule_digest: receipt.schedule.digest,
      candidate_index: receipt.candidate.index,
      choices: receipt.candidate.choices,
      status: receipt.status
    }
  end

  @doc "Encodes a receipt as JSON text."
  @spec encode!(t()) :: String.t()
  def encode!(%__MODULE__{} = receipt) do
    receipt |> to_map() |> JSON.encode!()
  end

  @doc "Decodes JSON text produced by `encode!/1`."
  @spec decode!(String.t()) :: t()
  def decode!(json) when is_binary(json) do
    json
    |> JSON.decode!()
    |> from_map!()
  end

  @doc "Renders a receipt as a JSON map."
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = receipt) do
    Map.from_struct(receipt)
  end

  @doc "Builds a receipt from a JSON-decoded map."
  @spec from_map!(map()) :: t()
  def from_map!(%{} = map) do
    map =
      map
      |> Map.new(fn {key, value} ->
        {String.to_atom(key), deep_atomize_keys(key, value)}
      end)
      |> Map.update!(:status, &deep_atomize_keys/1)

    struct!(__MODULE__, map)
  end

  # Candidate choices are user-provided data: their keys must survive JSON
  # round-trips unchanged so identity comparisons stay exact.
  defp deep_atomize_keys("choices", value), do: value

  defp deep_atomize_keys(_key, map) when is_map(map) do
    Map.new(map, fn {key, value} -> {String.to_atom(key), deep_atomize_keys(key, value)} end)
  end

  defp deep_atomize_keys(_key, list) when is_list(list),
    do: Enum.map(list, &deep_atomize_keys("", &1))

  defp deep_atomize_keys(_key, value) when value in ["ok", "evaluated", "failed"],
    do: String.to_atom(value)

  defp deep_atomize_keys(_key, value) when value in ["hit", "miss"],
    do: String.to_atom(value)

  defp deep_atomize_keys(_key, value), do: value

  defp deep_atomize_keys(map) when is_map(map) do
    Map.new(map, fn {key, value} -> {String.to_atom(key), deep_atomize_keys(key, value)} end)
  end

  defp deep_atomize_keys(list) when is_list(list), do: Enum.map(list, &deep_atomize_keys/1)

  defp deep_atomize_keys(value) when value in ["ok", "evaluated", "failed"],
    do: String.to_atom(value)

  defp deep_atomize_keys(value) when value in ["hit", "miss"], do: String.to_atom(value)
  defp deep_atomize_keys(value), do: value
end
