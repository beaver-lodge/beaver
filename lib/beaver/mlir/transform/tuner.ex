defmodule Beaver.MLIR.Transform.Tuner do
  @moduledoc """
  Bounded, deterministic BEAM evaluation of resolved Transform schedules.

  Candidate order is the order returned by `Schedule.enumerate/2`, independent
  of task completion order. Timeouts kill the individual task. Cancellation is
  shared through an explicit token and can also be observed by cooperative
  evaluator callbacks.

  Searches emit `[:beaver, :mlir, :compilation, :autotuning, :start | :stop]`
  and each candidate emits the corresponding
  `[:beaver, :mlir, :compilation, :autotuning, :candidate, :start | :stop]`
  events. Candidate metadata can be passed to `Beaver.MLIR.ActionTracing` to
  correlate lower-level MLIR actions with the schedule that caused them.
  """

  alias Beaver.MLIR
  alias MLIR.Transform
  alias Transform.Schedule

  defmodule Evaluator do
    @moduledoc "Optional behaviour for schedule benchmark/evaluation adapters."

    @callback evaluate(Schedule.Resolved.t(), context :: map(), state :: term()) ::
                term()
  end

  defmodule Cancellation do
    @moduledoc "An explicit, process-independent cancellation token."
    @enforce_keys [:state]
    defstruct [:state]

    @type t() :: %__MODULE__{state: reference()}

    @spec new() :: t()
    def new do
      state = :atomics.new(1, signed: false)
      %__MODULE__{state: state}
    end

    @spec cancel(t()) :: :ok
    def cancel(%__MODULE__{state: state}) do
      :ok = :atomics.put(state, 1, 1)
    end

    @spec cancelled?(t()) :: boolean()
    def cancelled?(%__MODULE__{state: state}), do: :atomics.get(state, 1) == 1
  end

  defmodule Candidate do
    @moduledoc "A deterministic candidate evaluation record."
    @enforce_keys [:index, :choices, :status, :duration]
    defstruct [:index, :choices, :status, :schedule, :value, :metadata, :reason, :duration]

    @type status() ::
            :ok
            | :evaluation_failure
            | :invalid_schedule
            | :constraint_failure
            | :timeout
            | :cancelled

    @type t() :: %__MODULE__{
            index: non_neg_integer(),
            choices: map(),
            status: status(),
            schedule: Schedule.Resolved.t() | nil,
            value: term(),
            metadata: term(),
            reason: term(),
            duration: integer()
          }
  end

  defmodule Result do
    @moduledoc "Ordered records and runtime configuration from one tuning search."
    @enforce_keys [:candidates, :cancellation, :max_concurrency, :timeout]
    defstruct [:candidates, :cancellation, :max_concurrency, :timeout]

    @type t() :: %__MODULE__{
            candidates: [Candidate.t()],
            cancellation: Cancellation.t(),
            max_concurrency: pos_integer(),
            timeout: non_neg_integer() | :infinity
          }
  end

  @type evaluator() :: function() | module() | {module(), term()}

  @doc """
  Resolves and evaluates every candidate with bounded concurrency.

  An evaluator may return a bare value, `{:ok, value}`,
  `{:ok, value, metadata}`, or `{:error, reason}`. Beaver records these values
  without imposing a scoring or winner-selection policy.
  """
  @spec search(Schedule.input(), evaluator(), keyword()) ::
          {:ok, Result.t()} | {:error, Transform.Error.t()}
  def search(schedule, evaluator, opts \\ []) do
    with :ok <- validate_options(opts),
         {:ok, {snapshot, sequence}} <- Schedule.snapshot(schedule, opts),
         {:ok, choices} <- candidate_choices({:bytecode, snapshot}, sequence, opts) do
      max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())
      timeout = Keyword.get(opts, :timeout, 60_000)
      cancellation = Keyword.get(opts, :cancellation, Cancellation.new())

      runtime = %{
        evaluator: evaluator_and_state(evaluator),
        cancellation: cancellation,
        solver: Keyword.get(opts, :solver),
        telemetry_opts: opts
      }

      started = System.monotonic_time()

      search_metadata = %{
        sequence: sequence,
        candidate_count: length(choices),
        max_concurrency: max_concurrency,
        timeout: timeout
      }

      MLIR.Telemetry.emit([:autotuning, :start], %{}, search_metadata, opts)

      records =
        evaluate_candidates(
          choices,
          snapshot,
          sequence,
          runtime,
          max_concurrency,
          timeout
        )

      result = %Result{
        candidates: records,
        cancellation: cancellation,
        max_concurrency: max_concurrency,
        timeout: timeout
      }

      MLIR.Telemetry.emit(
        [:autotuning, :stop],
        %{duration: System.monotonic_time() - started, candidate_count: length(records)},
        Map.put(search_metadata, :status_counts, Enum.frequencies_by(records, & &1.status)),
        opts
      )

      {:ok, result}
    end
  end

  defp evaluate_candidates(
         choices,
         snapshot,
         sequence,
         runtime,
         max_concurrency,
         timeout
       ) do
    indexed = Enum.with_index(choices, fn choices, index -> {index, choices} end)

    indexed
    |> Task.async_stream(
      &evaluate_candidate(&1, snapshot, sequence, runtime),
      max_concurrency: max_concurrency,
      ordered: true,
      timeout: timeout,
      on_timeout: :kill_task
    )
    |> Enum.zip(indexed)
    |> Enum.map(&candidate_record(&1, sequence, timeout, runtime.telemetry_opts))
  end

  defp candidate_record(
         {{:ok, %Candidate{} = candidate}, _indexed},
         _sequence,
         _timeout,
         _telemetry_opts
       ),
       do: candidate

  defp candidate_record(
         {{:exit, reason}, {index, choices}},
         sequence,
         timeout,
         telemetry_opts
       ) do
    status = if timeout_exit?(reason), do: :timeout, else: :evaluation_failure

    candidate = %Candidate{
      index: index,
      choices: choices,
      status: status,
      reason: if(status == :timeout, do: :timeout, else: reason),
      duration: timeout_duration(timeout)
    }

    emit_candidate_stop(candidate, sequence, telemetry_opts)
    candidate
  end

  @doc "Bang variant of `search/3`."
  @spec search!(Schedule.input(), evaluator(), keyword()) :: Result.t()
  def search!(schedule, evaluator, opts \\ []) do
    case search(schedule, evaluator, opts) do
      {:ok, result} -> result
      {:error, error} -> raise error
    end
  end

  @doc "Passes successful records to an application-defined selection function."
  @spec select(Result.t(), ([Candidate.t()] -> term())) :: term()
  def select(%Result{candidates: candidates}, selector) when is_function(selector, 1) do
    candidates
    |> Enum.filter(&(&1.status == :ok))
    |> selector.()
  end

  defp candidate_choices(schedule, sequence, opts) do
    case Keyword.fetch(opts, :candidates) do
      {:ok, candidates} when is_list(candidates) ->
        if Enum.all?(candidates, &is_map/1) do
          {:ok, candidates}
        else
          {:error,
           %Transform.Error{
             kind: :invalid_schedule,
             reason: {:invalid_candidates, candidates}
           }}
        end

      {:ok, candidates} ->
        {:error,
         %Transform.Error{
           kind: :invalid_schedule,
           reason: {:invalid_candidates, candidates}
         }}

      :error ->
        Schedule.enumerate(schedule,
          sequence: sequence,
          max_candidates: Keyword.get(opts, :max_candidates, 10_000)
        )
    end
  end

  defp evaluate_candidate(
         {index, choices},
         snapshot,
         sequence,
         runtime
       ) do
    started = System.monotonic_time()

    MLIR.Telemetry.emit(
      [:autotuning, :candidate, :start],
      %{},
      candidate_metadata(index, choices, sequence),
      runtime.telemetry_opts
    )

    candidate =
      try do
        if Cancellation.cancelled?(runtime.cancellation) do
          candidate(index, choices, :cancelled, started, reason: :cancelled)
        else
          resolve_and_evaluate(index, choices, snapshot, sequence, runtime, started)
        end
      rescue
        exception ->
          candidate(index, choices, :evaluation_failure, started,
            reason: Exception.format(:error, exception, __STACKTRACE__)
          )
      catch
        kind, reason ->
          candidate(index, choices, :evaluation_failure, started,
            reason: Exception.format(kind, reason, __STACKTRACE__)
          )
      end

    emit_candidate_stop(candidate, sequence, runtime.telemetry_opts)
    candidate
  end

  defp resolve_and_evaluate(
         index,
         choices,
         snapshot,
         sequence,
         runtime,
         started
       ) do
    case Schedule.resolve({:bytecode, snapshot}, choices,
           sequence: sequence,
           solver: runtime.solver
         ) do
      {:ok, resolved} ->
        continue_evaluation(index, choices, resolved, sequence, runtime, started)

      {:error, %Transform.Error{} = error} ->
        candidate(index, choices, error_status(error), started, reason: error)
    end
  end

  defp continue_evaluation(
         index,
         choices,
         resolved,
         sequence,
         runtime,
         started
       ) do
    if Cancellation.cancelled?(runtime.cancellation) do
      candidate(index, choices, :cancelled, started,
        schedule: resolved,
        reason: :cancelled
      )
    else
      evaluate_resolved(index, choices, resolved, sequence, runtime, started)
    end
  end

  defp evaluate_resolved(
         index,
         choices,
         resolved,
         sequence,
         runtime,
         started
       ) do
    context = %{
      index: index,
      choices: choices,
      cancellation: runtime.cancellation,
      cancelled?: fn -> Cancellation.cancelled?(runtime.cancellation) end,
      telemetry: Keyword.get(runtime.telemetry_opts, :telemetry),
      telemetry_metadata:
        candidate_metadata(index, choices, sequence)
        |> Map.put(:transform_schedule_digest, resolved.digest)
    }

    case call_evaluator(runtime.evaluator, resolved, context) do
      {:ok, value, metadata} ->
        candidate(index, choices, :ok, started,
          schedule: resolved,
          value: value,
          metadata: metadata
        )

      {:error, reason} ->
        candidate(index, choices, :evaluation_failure, started,
          schedule: resolved,
          reason: reason
        )
    end
  end

  defp evaluator_and_state({module, state}) when is_atom(module), do: {{:module, module}, state}
  defp evaluator_and_state(module) when is_atom(module), do: {{:module, module}, nil}
  defp evaluator_and_state(evaluator), do: {evaluator, nil}

  defp call_evaluator({evaluator, state}, resolved, _context) when is_function(evaluator, 1) do
    normalize_evaluation(evaluator.(resolved), state)
  end

  defp call_evaluator({evaluator, state}, resolved, context) when is_function(evaluator, 2) do
    normalize_evaluation(evaluator.(resolved, context), state)
  end

  defp call_evaluator({{:module, module}, state}, resolved, context) do
    normalize_evaluation(module.evaluate(resolved, context, state), state)
  end

  defp normalize_evaluation({:ok, value, metadata}, _state), do: {:ok, value, metadata}
  defp normalize_evaluation({:ok, value}, state), do: {:ok, value, state}
  defp normalize_evaluation({:error, reason}, _state), do: {:error, reason}
  defp normalize_evaluation(value, state), do: {:ok, value, state}

  defp candidate(index, choices, status, started, fields) do
    struct!(
      Candidate,
      Keyword.merge(
        [
          index: index,
          choices: choices,
          status: status,
          duration: System.monotonic_time() - started
        ],
        fields
      )
    )
  end

  defp error_status(%Transform.Error{kind: :constraint_failure}), do: :constraint_failure
  defp error_status(%Transform.Error{}), do: :invalid_schedule

  defp timeout_exit?(:timeout), do: true
  defp timeout_exit?({:timeout, _}), do: true
  defp timeout_exit?(_reason), do: false

  defp timeout_duration(:infinity), do: 0
  defp timeout_duration(timeout), do: System.convert_time_unit(timeout, :millisecond, :native)

  defp candidate_metadata(index, choices, sequence) do
    %{candidate_index: index, choices: choices, sequence: sequence}
  end

  defp emit_candidate_stop(candidate, sequence, telemetry_opts) do
    metadata =
      candidate_metadata(candidate.index, candidate.choices, sequence)
      |> Map.put(:status, candidate.status)
      |> maybe_put_schedule_digest(candidate.schedule)

    MLIR.Telemetry.emit(
      [:autotuning, :candidate, :stop],
      %{duration: candidate.duration},
      metadata,
      telemetry_opts
    )
  end

  defp maybe_put_schedule_digest(metadata, %Schedule.Resolved{digest: digest}),
    do: Map.put(metadata, :transform_schedule_digest, digest)

  defp maybe_put_schedule_digest(metadata, _schedule), do: metadata

  defp validate_options(opts) do
    if Keyword.keyword?(opts) do
      with :ok <- validate_supported_options(opts),
           :ok <- validate_max_concurrency(opts),
           :ok <- validate_timeout(opts),
           :ok <- validate_cancellation(opts) do
        validate_telemetry(opts)
      end
    else
      invalid_options("tuner options must be a keyword list")
    end
  end

  defp validate_supported_options(opts) do
    supported = [
      :sequence,
      :ctx,
      :candidates,
      :max_candidates,
      :max_concurrency,
      :timeout,
      :cancellation,
      :solver,
      :telemetry
    ]

    case Keyword.keys(opts) -- supported do
      [] -> :ok
      unsupported -> invalid_options("unsupported tuner options: #{inspect(unsupported)}")
    end
  end

  defp validate_max_concurrency(opts) do
    case Keyword.get(opts, :max_concurrency, System.schedulers_online()) do
      value when is_integer(value) and value > 0 -> :ok
      _value -> invalid_options(":max_concurrency must be a positive integer")
    end
  end

  defp validate_timeout(opts) do
    case Keyword.get(opts, :timeout, 60_000) do
      :infinity -> :ok
      value when is_integer(value) and value >= 0 -> :ok
      _value -> invalid_options(":timeout must be a non-negative integer or :infinity")
    end
  end

  defp validate_cancellation(opts) do
    case Keyword.get(opts, :cancellation) do
      nil ->
        :ok

      %Cancellation{} ->
        :ok

      value ->
        invalid_options(":cancellation must be a Cancellation token, got: #{inspect(value)}")
    end
  end

  defp validate_telemetry(opts) do
    case Keyword.get(opts, :telemetry) do
      nil -> :ok
      callback when is_function(callback, 3) -> :ok
      _value -> invalid_options(":telemetry must be a three-argument function")
    end
  end

  defp invalid_options(reason) do
    {:error, %Transform.Error{kind: :invalid_schedule, reason: reason}}
  end
end
