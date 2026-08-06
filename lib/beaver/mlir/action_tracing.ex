defmodule Beaver.MLIR.ActionTracing do
  @moduledoc """
  Exposes MLIR Action Tracing through Elixir telemetry.

  MLIR dispatches actions — pass execution, rewrite-pattern application,
  tiling, and other compiler steps — through an action handler registered on
  an `MLIR.Context`. This module attaches a context-scoped observer that
  records `before`/`after` action events, hands them to the BEAM as structured
  telemetry, and can skip or limit actions by tag.

  Events are drained explicitly (or on a timer) and emitted as telemetry:

  - `[:beaver, :mlir, :compilation, :action, :start]` — before an action runs
  - `[:beaver, :mlir, :compilation, :action, :stop]` — after an action runs

  Each event carries `tag`, `description`, `depth`, and `ir_units` metadata.
  The `stop` event carries a `duration` measurement computed from the paired
  `start` event.

  Native observers run on MLIR worker threads and never invoke BEAM APIs
  directly; events are queued on the native side and drained by the BEAM.

  ## Options

  - `:tags` — only observe actions whose tag is in this list. Defaults to all.
  - `:locations` — only observe actions whose context IR units carry a
    matching source location substring. Defaults to all.
  - `:skip` — map of tag to a non-negative skip count; the first N occurrences
    of that tag are skipped (not executed).
  - `:limit` — map of tag to a non-negative execution limit; further
    occurrences are skipped once the limit is reached.
  - `:drain_interval_ms` — when set, a periodic drainer emits telemetry
    automatically. Defaults to `nil` (manual draining).
  - `:telemetry` — optional `(event, measurements, metadata) -> term` callback
    used instead of `:telemetry` events (see `Beaver.MLIR.Telemetry`).
  - `:metadata` — metadata merged into every emitted action event. Event-owned
    fields take precedence. This can correlate actions with a higher-level
    operation such as one Transform autotuning candidate.
  """

  use GenServer

  alias Beaver.MLIR

  @registry __MODULE__.Registry

  defmodule Session do
    @moduledoc "A live context-scoped action tracing session."
    @enforce_keys [:pid, :context]
    defstruct [:pid, :context, :tags, :locations, :skip, :limit, :drain_interval_ms, :metadata]

    @type t() :: %__MODULE__{
            pid: pid(),
            context: MLIR.Context.t(),
            tags: [String.t()] | nil,
            locations: [String.t()] | nil,
            skip: %{optional(String.t()) => non_neg_integer()},
            limit: %{optional(String.t()) => non_neg_integer()},
            drain_interval_ms: pos_integer() | nil,
            metadata: map()
          }
  end

  @doc false
  def global_registrar_child_specs do
    [{Registry, keys: :duplicate, name: @registry}]
  end

  @doc """
  Attaches an action tracing session to `context`.

  Returns a `Session` holding the native session resource. Call `drain/1`
  periodically (or pass `drain_interval_ms`) to receive events, and
  `detach/1` when done. The session is detached automatically when the context
  is destroyed.
  """
  @spec attach(MLIR.Context.t(), keyword()) :: Session.t()
  def attach(%MLIR.Context{} = context, opts \\ []) do
    tags = normalize_tags!(Keyword.get(opts, :tags))
    locations = normalize_locations!(Keyword.get(opts, :locations))
    skip = normalize_count_map!(Keyword.get(opts, :skip, %{}), :skip)
    limit = normalize_count_map!(Keyword.get(opts, :limit, %{}), :limit)
    metadata = normalize_metadata!(Keyword.get(opts, :metadata, %{}))

    drain_interval_ms =
      case Keyword.get(opts, :drain_interval_ms) do
        nil -> nil
        ms when is_integer(ms) and ms > 0 -> ms
        other -> raise ArgumentError, "invalid :drain_interval_ms: #{inspect(other)}"
      end

    filter_json = "[" <> Enum.map_join(tags || [], ",", &JSON.encode!/1) <> "]"
    location_json = "[" <> Enum.map_join(locations || [], ",", &JSON.encode!/1) <> "]"

    skip_json = encode_count_map(skip)
    limit_json = encode_count_map(limit)

    session =
      MLIR.CAPI.beaver_raw_action_tracing_attach(
        context.ref,
        filter_json,
        location_json,
        skip_json,
        limit_json
      )

    {:ok, pid} =
      GenServer.start(__MODULE__, %{
        context: context,
        session: session,
        telemetry: Keyword.get(opts, :telemetry),
        metadata: metadata,
        drain_interval_ms: drain_interval_ms
      })

    %Session{
      pid: pid,
      context: context,
      tags: tags,
      locations: locations,
      skip: skip,
      limit: limit,
      drain_interval_ms: drain_interval_ms,
      metadata: metadata
    }
  end

  @doc """
  Drains pending action events from the session and emits telemetry.
  Returns the list of decoded events.
  """
  @spec drain(Session.t() | pid()) :: [map()]
  def drain(%Session{} = session), do: drain(session.pid)
  def drain(pid) when is_pid(pid), do: GenServer.call(pid, :drain, 30_000)

  @doc "Detaches and releases the tracing session."
  @spec detach(Session.t() | pid()) :: :ok
  def detach(%Session{} = session), do: detach(session.pid)
  def detach(pid) when is_pid(pid), do: GenServer.call(pid, :detach, 5_000)

  @doc false
  def release_context(%MLIR.Context{ref: context_ref}) do
    if Process.whereis(@registry) do
      Registry.dispatch(@registry, context_ref, fn entries ->
        Enum.each(entries, fn {pid, nil} ->
          try do
            GenServer.call(pid, :context_destroyed, 5_000)
          catch
            :exit, {:noproc, _} -> :ok
            :exit, {:normal, _} -> :ok
          end
        end)
      end)
    end

    :ok
  end

  @impl true
  def init(state) do
    {:ok, _} = Registry.register(@registry, state.context.ref, nil)

    if state.drain_interval_ms,
      do: Process.send_after(self(), :drain_timer, state.drain_interval_ms)

    {:ok, state}
  end

  @impl true
  def handle_call(:drain, _from, state) do
    events = drain_raw_events(state)
    emit_events(events, state.telemetry, state.metadata)
    {:reply, events, state}
  end

  def handle_call(:detach, _from, state) do
    detach_session(state)
    {:stop, :normal, :ok, state}
  end

  def handle_call(:context_destroyed, _from, state) do
    detach_session(state)
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_info(:drain_timer, state) do
    events = drain_raw_events(state)
    emit_events(events, state.telemetry, state.metadata)
    schedule_drain(state)
    {:noreply, state}
  end

  defp schedule_drain(%{drain_interval_ms: nil}), do: :ok

  defp schedule_drain(%{drain_interval_ms: ms}) do
    Process.send_after(self(), :drain_timer, ms)
  end

  defp drain_raw_events(state) do
    state.session
    |> MLIR.CAPI.beaver_raw_action_tracing_drain()
    |> JSON.decode!()
  end

  defp detach_session(state) do
    try do
      MLIR.CAPI.beaver_raw_action_tracing_detach(state.session)
    catch
      :exit, _ -> :ok
    end

    :ok
  end

  # Pair before/after events in order; compute the after event's duration.
  defp pair_events(events) when is_list(events) do
    {paired, _open} =
      Enum.reduce(events, {[], %{}}, fn
        %{"phase" => "before"} = event, {acc, open} ->
          key = event_key(event)
          {acc, Map.put(open, key, event)}

        %{"phase" => "after"} = event, {acc, open} ->
          key = event_key(event)
          start = Map.get(open, key)
          {[{:stop, start, event, duration(event, start)} | acc], Map.delete(open, key)}

        other, {acc, open} ->
          {[{:bare, other} | acc], open}
      end)

    Enum.reverse(paired)
  end

  defp pair_events(other), do: raise("invalid action tracing event payload: #{inspect(other)}")

  defp duration(event, %{"t_ns" => before_ns}) do
    case event do
      %{"t_ns" => after_ns} when is_integer(after_ns) and is_integer(before_ns) ->
        max(after_ns - before_ns, 0)

      _ ->
        nil
    end
  end

  defp duration(_event, _start), do: nil

  defp event_key(event) do
    {event["tag"], event["depth"]}
  end

  defp emit_events(events, telemetry, extra_metadata) do
    events
    |> pair_events()
    |> Enum.each(fn
      {:stop, start, event, duration} ->
        start_event = start || %{"tag" => event["tag"], "depth" => event["depth"]}

        MLIR.Telemetry.emit(
          [:action, :start],
          %{},
          Map.merge(extra_metadata, start_event),
          telemetry: telemetry
        )

        MLIR.Telemetry.emit(
          [:action, :stop],
          %{duration: duration},
          Map.merge(extra_metadata, %{
            "tag" => event["tag"],
            "depth" => event["depth"]
          }),
          telemetry: telemetry
        )

      {:bare, _event} ->
        :ok
    end)

    :ok
  end

  defp normalize_tags!(nil), do: nil

  defp normalize_tags!(tags) when is_list(tags) do
    unless Enum.all?(tags, &is_binary/1) do
      raise ArgumentError, ":tags must be a list of strings"
    end

    tags
  end

  defp normalize_tags!(other), do: raise(ArgumentError, "invalid :tags: #{inspect(other)}")

  defp normalize_locations!(nil), do: nil

  defp normalize_locations!(locations) when is_list(locations) do
    unless Enum.all?(locations, &is_binary/1) do
      raise ArgumentError, ":locations must be a list of strings"
    end

    locations
  end

  defp normalize_locations!(other),
    do: raise(ArgumentError, "invalid :locations: #{inspect(other)}")

  defp normalize_count_map!(map, _name) when is_map(map) do
    unless Enum.all?(map, fn {k, v} -> is_binary(k) and is_integer(v) and v >= 0 end) do
      raise ArgumentError, "count map values must be non-negative integers"
    end

    map
  end

  defp normalize_count_map!(other, name),
    do: raise(ArgumentError, "invalid #{name}: #{inspect(other)}")

  defp normalize_metadata!(metadata) when is_map(metadata), do: metadata

  defp normalize_metadata!(other),
    do: raise(ArgumentError, "invalid :metadata: #{inspect(other)}")

  defp encode_count_map(map) do
    "{" <> Enum.map_join(map, ",", fn {k, v} -> ~s("#{k}":#{v}) end) <> "}"
  end
end
