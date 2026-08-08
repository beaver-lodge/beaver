defmodule Beaver.Shadow.Tuning do
  @moduledoc """
  Tuning receipts for Triton-style config spaces.

  This is the Shadow Wavefront answer to the Triton autotuner
  infrastructure gaps (triton-lang/triton#4020, #11174, vllm#37188,
  pytorch#186886): an external reference schema that makes three things
  first-class:

  1. **decision/observation separation** — a tuning record's identity only
     contains stable facts (kernel digest, target, config space digest,
     config, winner); durations never enter it, which is the precondition for
     portable, replayable cache keys;
  2. **failure as data** — per-config failures are recorded structurally
     (kind + reason), so a crashing config is evidence instead of a process
     kill (see `Beaver.Shadow.Probe.probe_many/2` for the isolated execution);
  3. **structural proxy auditability** — `ttg.convert_layout` counts and
     lowering capability are standard record fields, so pruning decisions can
     be audited and regressed without a GPU.

  The record is deliberately CPU-only: `run/3` evaluates every config through
  `Beaver.Shadow.OptimizationTrial` without launching kernels.
  """

  alias Beaver.MLIR
  alias Beaver.Shadow.Corpus
  alias Beaver.Shadow.OptimizationTrial
  alias Beaver.Shadow.Tuning.GPU

  @format 1

  defmodule Config do
    @moduledoc "One tuning candidate, mirroring the stable parts of `triton.Config`."
    @enforce_keys [:index, :num_warps]
    defstruct [:index, :num_warps, :num_stages, :num_ctas]

    @type t() :: %__MODULE__{
            index: non_neg_integer(),
            num_warps: pos_integer(),
            num_stages: pos_integer() | nil,
            num_ctas: pos_integer() | nil
          }

    @doc "Stable digest of one config (does not depend on observations)."
    @spec digest(t()) :: String.t()
    def digest(%__MODULE__{} = config) do
      config
      |> Map.from_struct()
      |> :erlang.term_to_binary()
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)
    end
  end

  defmodule Record do
    @moduledoc "One tuning evaluation: provenance + decision + observations."
    @enforce_keys [:format, :kernel_digest, :config_space_digest, :config, :status]
    defstruct [
      :format,
      :kernel_digest,
      :target,
      :capability,
      :config_space_digest,
      :config,
      :status,
      :failure,
      :structural_proxy,
      :timings
    ]

    @type t() :: %__MODULE__{
            format: pos_integer(),
            kernel_digest: String.t(),
            target: String.t() | nil,
            capability: term() | nil,
            config_space_digest: String.t(),
            config: Config.t(),
            status: :evaluated | :failed,
            failure: map() | nil,
            structural_proxy: map() | nil,
            timings: map() | nil
          }
  end

  defmodule Run do
    @moduledoc "One tuning pass over a config space."
    @enforce_keys [:fixture, :kernel_digest, :records, :winner]
    defstruct [:fixture, :kernel_digest, :records, :winner]

    @type t() :: %__MODULE__{
            fixture: atom(),
            kernel_digest: String.t(),
            records: [Record.t()],
            winner: Record.t() | nil
          }
  end

  defmodule Prune do
    @moduledoc "A structural pruning decision with its audit trail."
    @enforce_keys [:fixture, :config_space_digest, :decisions, :kept]
    defstruct [:fixture, :config_space_digest, :decisions, :kept]

    @type t() :: %__MODULE__{
            fixture: atom(),
            config_space_digest: String.t(),
            decisions: [map()],
            kept: [Config.t()]
          }
  end

  defmodule Event do
    @moduledoc """
    A JSON event isomorphic to triton's `AutotuneListener` protocol.

    The six upstream fields (`fn`, `key`, `best_config`, `configs_timings`,
    `duration`, `cache_hit`) map one-to-one; `structural_proxy` and `failure`
    are the extensions the upstream protocol is missing.
    """
    @enforce_keys [:fn, :key, :best_config, :configs_timings, :cache_hit]
    defstruct [
      :fn,
      :key,
      :best_config,
      :configs_timings,
      :duration,
      :cache_hit,
      :structural_proxy,
      :failure
    ]

    @type t() :: %__MODULE__{
            fn: String.t(),
            key: term(),
            best_config: Config.t() | nil,
            configs_timings: [{Config.t(), [non_neg_integer()]}],
            duration: non_neg_integer() | nil,
            cache_hit: boolean(),
            structural_proxy: map() | nil,
            failure: map() | nil
          }
  end

  @doc "Current tuning record schema version."
  def format, do: @format

  @doc """
  Runs a CPU-only tuning pass over `configs` for a corpus fixture.

  Every config is evaluated through `OptimizationTrial` (baseline/optimized
  `ttg.convert_layout` counts + lowering capability), recorded with its
  config-space provenance, and the winner is picked by the structural proxy
  (lowering capable first, then fewest optimized conversions). Durations are
  observations and never enter `identity/1`.
  """
  @spec run(atom(), [Config.t()], keyword()) :: Run.t()
  def run(fixture_name, configs, opts \\ []) when is_atom(fixture_name) do
    fixture = Corpus.fixture(fixture_name)
    target = Keyword.get(opts, :target, "cuda:80")
    gpu? = Keyword.get(opts, :gpu, false)
    launch = Keyword.get(opts, :launch, Map.get(fixture, :launch))
    config_space_digest = configs_digest(configs)
    kernel_digest = kernel_digest(fixture)

    context = MLIR.Context.create(all_dialects: false)
    on_exit = Keyword.get(opts, :on_exit)

    records =
      try do
        Beaver.Triton.register(context)

        Enum.map(configs, fn config ->
          evaluate(
            fixture,
            config,
            context,
            target,
            config_space_digest,
            kernel_digest,
            gpu?,
            launch
          )
        end)
      after
        MLIR.Context.destroy(context)
        if is_function(on_exit), do: on_exit.()
      end

    %Run{
      fixture: fixture.name,
      kernel_digest: kernel_digest,
      records: records,
      winner: pick_winner(records)
    }
  end

  @doc "Stable identity of a tuning record; observations are excluded."
  @spec identity(Record.t()) :: String.t()
  def identity(%Record{} = record) do
    record
    |> stable_facts()
    |> then(&:crypto.hash(:sha256, :erlang.term_to_binary(&1)))
    |> Base.encode16(case: :lower)
  end

  @doc "The identity-bearing fields of a tuning record."
  @spec stable_facts(Record.t()) :: map()
  def stable_facts(%Record{} = record) do
    %{
      format: record.format,
      kernel_digest: record.kernel_digest,
      target: record.target,
      capability: record.capability,
      config_space_digest: record.config_space_digest,
      config_index: record.config.index,
      config_digest: Config.digest(record.config),
      status: record.status,
      failure_kind: if(record.failure, do: record.failure.kind, else: nil),
      structural_proxy: record.structural_proxy
    }
  end

  @doc "Stable digest of a config space (ordered, index-independent)."
  @spec configs_digest([Config.t()]) :: String.t()
  def configs_digest(configs) when is_list(configs) do
    configs
    |> Enum.map(&{&1.num_warps, &1.num_stages, &1.num_ctas})
    |> :erlang.term_to_binary()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  @doc "Picks the winner by the structural proxy: lowering capable, fewest conversions."
  @spec pick_winner([Record.t()]) :: Record.t() | nil
  def pick_winner(records) do
    records
    |> Enum.filter(&(&1.status == :evaluated))
    |> Enum.sort_by(fn record ->
      proxy = record.structural_proxy
      {not proxy.lowered_to_llvm, proxy.optimized}
    end)
    |> List.first()
  end

  @doc """
  Picks the winner by measured GPU latency (lowest median first).
  """
  @spec pick_winner_by_latency([Record.t()]) :: Record.t() | nil
  def pick_winner_by_latency(records) do
    records
    |> Enum.filter(&(&1.status == :evaluated and is_number(&1.timings && &1.timings.gpu_ns)))
    |> Enum.sort_by(& &1.timings.gpu_ns)
    |> List.first()
  end

  @doc """
  Spearman rank correlation between a structural fact and GPU latency.

  `proxy_fun` extracts the structural value and `latency_fun` the latency from
  each record; only records with both are used. Returns `nil` when fewer than
  two points are available.
  """
  @spec correlation([Record.t()], (Record.t() -> number() | nil), (Record.t() -> number() | nil)) ::
          float() | nil
  def correlation(records, proxy_fun, latency_fun) do
    pairs =
      records
      |> Enum.map(fn record ->
        {proxy_fun.(record), latency_fun.(record)}
      end)
      |> Enum.filter(fn {proxy, latency} -> is_number(proxy) and is_number(latency) end)

    if length(pairs) < 2 do
      nil
    else
      {proxy_ranks, _} = rank(pairs |> Enum.map(&elem(&1, 0)))
      {latency_ranks, _} = rank(pairs |> Enum.map(&elem(&1, 1)))
      pearson(proxy_ranks, latency_ranks)
    end
  end

  @doc """
  Prunes a config space with the structural proxy and returns an audit trail.

  Configs are ranked by lowering capability first and optimized
  `ttg.convert_layout` count second; the top `:max_keep` are kept. Every
  decision records the structural facts that drove it, so the pruning is
  auditable and regressable without a GPU.
  """
  @spec prune_by_structure(atom() | Run.t(), [Config.t()], keyword()) :: Prune.t()
  def prune_by_structure(fixture_or_run, configs, opts \\ [])

  def prune_by_structure(%Run{} = run, _configs, opts) do
    max_keep = Keyword.get(opts, :max_keep, 2)
    decisions = rank_decisions(run.records, max_keep)

    %Prune{
      fixture: run.fixture,
      config_space_digest:
        run.records
        |> List.first()
        |> then(&(&1 && &1.config_space_digest)),
      decisions: decisions,
      kept: decisions |> Enum.filter(& &1.keep) |> Enum.map(& &1.config)
    }
  end

  def prune_by_structure(fixture_name, configs, opts) when is_atom(fixture_name) do
    fixture_name
    |> run(configs)
    |> prune_by_structure(configs, opts)
  end

  @doc "Encodes a tuning run as JSON text."
  @spec encode!(Run.t()) :: String.t()
  def encode!(%Run{} = run) do
    %{
      "format" => @format,
      "fixture" => run.fixture,
      "kernel_digest" => run.kernel_digest,
      "records" => Enum.map(run.records, &record_to_map/1),
      "winner_index" => if(run.winner, do: run.winner.config.index, else: nil)
    }
    |> JSON.encode!()
  end

  @doc "Decodes JSON text produced by `encode!/1`."
  @spec decode!(String.t()) :: Run.t()
  def decode!(json) when is_binary(json) do
    map = JSON.decode!(json)

    records =
      Enum.map(map["records"], fn record_map ->
        record_map
        |> Map.new(fn
          {"config", value} -> {:config, config_from_map(value)}
          {"status", value} -> {:status, String.to_existing_atom(value)}
          {key, value} -> {String.to_existing_atom(key), deep_atomize(value)}
        end)
        |> then(&struct!(Record, &1))
      end)

    winner =
      Enum.find(records, &(&1.config.index == map["winner_index"]))

    %Run{
      fixture: map["fixture"] |> String.to_existing_atom(),
      kernel_digest: map["kernel_digest"],
      records: records,
      winner: winner
    }
  end

  @doc "Builds an AutotuneListener-isomorphic event from a tuning run."
  @spec event(Run.t(), keyword()) :: Event.t()
  def event(%Run{} = run, opts \\ []) do
    fn_name = Keyword.get(opts, :fn_name, "corpus_kernel")

    %Event{
      fn: fn_name,
      key:
        {run.kernel_digest, run.records |> List.first() |> then(&(&1 && &1.config_space_digest))},
      best_config: if(run.winner, do: run.winner.config, else: nil),
      configs_timings:
        Enum.map(run.records, fn record ->
          {record.config, if(record.timings, do: [record.timings.total_ns], else: [])}
        end),
      duration: total_duration(run.records),
      cache_hit: false,
      structural_proxy:
        run.records
        |> Enum.filter(& &1.structural_proxy)
        |> Map.new(fn record -> {record.config.index, record.structural_proxy} end),
      failure:
        run.records
        |> Enum.filter(&(&1.status == :failed))
        |> Enum.map(fn record ->
          %{config_index: record.config.index, failure: record.failure}
        end)
    }
  end

  defp evaluate(
         fixture,
         config,
         context,
         target,
         config_space_digest,
         kernel_digest,
         gpu?,
         launch
       ) do
    started = System.monotonic_time()
    source_text = File.read!(Corpus.fixture_path(fixture.name))

    try do
      module = MLIR.Module.create!(source_text, ctx: context)

      result =
        OptimizationTrial.run(module,
          target: target,
          num_warps: config.num_warps,
          gpu: false
        )

      gpu_latency =
        if gpu? and launch do
          case GPU.evaluate(source_text, context, target, config.num_warps, launch) do
            {:ok, ns} -> ns
            {:error, reason} -> reason
          end
        end

      proxy = %{
        baseline: result.baseline,
        optimized: result.optimized,
        reduction: result.baseline - result.optimized,
        lowered_to_llvm: result.lowered_to_llvm
      }

      %Record{
        format: @format,
        kernel_digest: kernel_digest,
        target: target,
        capability: nil,
        config_space_digest: config_space_digest,
        config: config,
        status: if(result.lowered_to_llvm, do: :evaluated, else: :failed),
        # GPU latency is an observation, never part of identity/1.
        timings: %{
          total_ns: System.monotonic_time() - started,
          gpu_ns: gpu_latency
        },
        failure:
          if(result.lowered_to_llvm,
            do: nil,
            else: %{kind: :lowering_failed, reason: "no llvm.func produced"}
          ),
        structural_proxy: proxy
      }
    rescue
      exception ->
        %Record{
          format: @format,
          kernel_digest: kernel_digest,
          target: target,
          capability: nil,
          config_space_digest: config_space_digest,
          config: config,
          status: :failed,
          failure: %{kind: :evaluation_error, reason: Exception.message(exception)},
          structural_proxy: nil,
          timings: %{total_ns: System.monotonic_time() - started}
        }
    end
  end

  defp rank(values) do
    sorted = values |> Enum.sort() |> Enum.with_index()

    rank_map =
      sorted |> Enum.group_by(&elem(&1, 0)) |> Map.new(fn {v, idxs} -> {v, avg_rank(idxs)} end)

    {Enum.map(values, &Map.fetch!(rank_map, &1)), sorted}
  end

  defp avg_rank(idxs) do
    Enum.map(idxs, &(elem(&1, 1) + 1)) |> Enum.sum() |> Kernel./(length(idxs))
  end

  defp pearson(xs, ys) do
    n = length(xs)
    x_mean = Enum.sum(xs) / n
    y_mean = Enum.sum(ys) / n

    {sxy, sxx, syy} =
      xs
      |> Enum.zip(ys)
      |> Enum.reduce({0.0, 0.0, 0.0}, fn {x, y}, {sxy, sxx, syy} ->
        dx = x - x_mean
        dy = y - y_mean
        {sxy + dx * dy, sxx + dx * dx, syy + dy * dy}
      end)

    if sxx == 0.0 or syy == 0.0, do: nil, else: sxy / :math.sqrt(sxx * syy)
  end

  defp kernel_digest(fixture) do
    fixture
    |> Map.fetch!(:name)
    |> Corpus.fixture_path()
    |> File.read!()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp total_duration(records) do
    records
    |> Enum.map(& &1.timings)
    |> Enum.reject(&is_nil/1)
    |> Enum.map(& &1.total_ns)
    |> Enum.sum()
    |> case do
      0 -> nil
      total -> total
    end
  end

  defp rank_decisions(records, max_keep) do
    records
    |> Enum.sort_by(fn record ->
      proxy = record.structural_proxy

      case proxy do
        nil -> {true, :infinity, record.config.index}
        proxy -> {not proxy.lowered_to_llvm, proxy.optimized, record.config.index}
      end
    end)
    |> Enum.with_index()
    |> Enum.map(fn {record, rank} ->
      %{
        config: record.config,
        structural_proxy: record.structural_proxy,
        status: record.status,
        rank: rank,
        keep: rank < max_keep,
        reason: prune_reason(record, rank, max_keep)
      }
    end)
  end

  defp prune_reason(record, rank, max_keep) when rank < max_keep do
    case record.structural_proxy do
      %{lowered_to_llvm: true, optimized: optimized} ->
        "rank #{rank}: lowers to LLVM, #{optimized} convert_layouts"

      nil ->
        "rank #{rank}: no structural facts (failed)"

      %{lowered_to_llvm: false} ->
        "rank #{rank}: does not lower to LLVM"
    end
  end

  defp prune_reason(_record, rank, max_keep) do
    "pruned: rank #{rank} >= max_keep #{max_keep}"
  end

  defp record_to_map(record) do
    record
    |> Map.from_struct()
    |> Map.update!(:config, &Map.from_struct/1)
  end

  defp config_from_map(map) do
    map
    |> Map.new(fn {key, value} -> {String.to_existing_atom(key), value} end)
    |> then(&struct!(Config, &1))
  end

  defp deep_atomize(map) when is_map(map) do
    Map.new(map, fn {key, value} ->
      {String.to_existing_atom(key), deep_atomize(value)}
    end)
  end

  defp deep_atomize(list) when is_list(list), do: Enum.map(list, &deep_atomize/1)
  defp deep_atomize(value), do: value
end
