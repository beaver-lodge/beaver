alias Beaver.MLIR.{CAPI, Context}

defmodule Beaver.MLIR.ActionTracing do
  @moduledoc """
  Context-scoped MLIR action tracing. Native observers queue events without
  calling BEAM; `drain/1` (or a timer) emits paired telemetry events under
  `[:beaver, :mlir, :compilation, :action, :start | :stop]`. Metadata contains
  action tags, descriptions, depth and IR units; stop measurements include duration.

  Options:
  - `:tags`, `:locations`: optional lists of tags or source-location substrings.
  - `:skip`, `:limit`: maps from tag to non-negative skipped/executed action counts.
  - `:drain_interval_ms`: a positive timer interval, or nil for manual draining.
  - `:telemetry`: optional `(event, measurements, metadata) -> term` callback.
  - `:metadata`: extra event metadata; event-owned fields take precedence.

  Sessions detach automatically when their context is destroyed.
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
            context: Context.t(),
            tags: [String.t()] | nil,
            locations: [String.t()] | nil,
            skip: %{optional(String.t()) => non_neg_integer()},
            limit: %{optional(String.t()) => non_neg_integer()},
            drain_interval_ms: pos_integer() | nil,
            metadata: map()
          }
  end

  @doc false
  def global_registrar_child_specs, do: [{Registry, keys: :duplicate, name: @registry}]

  @doc """
  Attaches a session to `context`. Call `drain/1` periodically (or supply `:drain_interval_ms`) to receive events,
  and `detach/1` when done.
  """
  @spec attach(Context.t(), keyword()) :: Session.t()
  def attach(%Context{} = ctx, opts \\ []) do
    filters = %{
      tags: normalize_strings!(opts[:tags], :tags),
      locations: normalize_strings!(opts[:locations], :locations),
      skip: normalize_count_map!(Keyword.get(opts, :skip, %{}), :skip),
      limit: normalize_count_map!(Keyword.get(opts, :limit, %{}), :limit),
      metadata: normalize_metadata!(Keyword.get(opts, :metadata, %{}))
    }

    drain_interval_ms =
      case Keyword.get(opts, :drain_interval_ms) do
        nil -> nil
        ms when is_integer(ms) and ms > 0 -> ms
        other -> raise ArgumentError, "invalid :drain_interval_ms: #{inspect(other)}"
      end

    json = Enum.map([:tags, :locations, :skip, :limit], &JSON.encode!(filters[&1] || []))
    session = apply(CAPI, :beaver_raw_action_tracing_attach, [ctx.ref | json])
    fields = Map.merge(filters, %{context: ctx, drain_interval_ms: drain_interval_ms})
    state = Map.merge(fields, %{session: session, telemetry: opts[:telemetry]})
    {:ok, pid} = GenServer.start(__MODULE__, state)
    struct!(Session, Map.put(fields, :pid, pid))
  end

  @doc "Drains pending action events from the session and emits telemetry. Returns the list of decoded events."
  @spec drain(Session.t() | pid()) :: [map()]
  def drain(%Session{} = session), do: drain(session.pid)
  def drain(pid) when is_pid(pid), do: GenServer.call(pid, :drain, 30_000)

  @doc "Detaches and releases the tracing session."
  @spec detach(Session.t() | pid()) :: :ok
  def detach(%Session{} = session), do: detach(session.pid)
  def detach(pid) when is_pid(pid), do: GenServer.call(pid, :detach, 5_000)

  @doc false
  def release_context(ctx), do: MLIR.ExternalInterface.release_context(ctx, @registry)

  @impl true
  def init(state) do
    {:ok, _} = Registry.register(@registry, state.context.ref, :attachment)
    if state.drain_interval_ms, do: Process.send_after(self(), :drain_timer, state.drain_interval_ms)
    {:ok, state}
  end

  @impl true
  def handle_call(:drain, _from, state) do
    events = drain_raw_events(state)
    emit_events(events, state.telemetry, state.metadata)
    {:reply, events, state}
  end

  def handle_call(command, _from, state) when command in [:detach, :context_destroyed] do
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

  defp schedule_drain(%{drain_interval_ms: ms}), do: Process.send_after(self(), :drain_timer, ms)

  defp drain_raw_events(state), do: state.session |> CAPI.beaver_raw_action_tracing_drain() |> JSON.decode!()

  defp detach_session(state) do
    CAPI.beaver_raw_action_tracing_detach(state.session)
    :ok
  catch
    :exit, _ -> :ok
  end

  # Retain only unmatched starts while emitting completed pairs in event order.
  defp emit_events(events, telemetry, extra_metadata) when is_list(events) do
    Enum.reduce(events, %{}, fn
      %{"phase" => "before"} = event, open ->
        Map.put(open, event_key(event), event)

      %{"phase" => "after"} = event, open ->
        {start, open} = Map.pop(open, event_key(event))
        identity = %{"tag" => event["tag"], "depth" => event["depth"]}
        MLIR.Telemetry.emit([:action, :start], %{}, Map.merge(extra_metadata, start || identity), telemetry: telemetry)

        MLIR.Telemetry.emit([:action, :stop], %{duration: duration(event, start)}, Map.merge(extra_metadata, identity),
          telemetry: telemetry
        )

        open

      _other, open ->
        open
    end)
  end

  defp emit_events(other, _, _), do: raise("invalid action tracing event payload: #{inspect(other)}")
  defp event_key(event), do: {event["tag"], event["depth"]}

  defp duration(%{"t_ns" => after_ns}, %{"t_ns" => before_ns})
       when is_integer(after_ns) and is_integer(before_ns), do: max(after_ns - before_ns, 0)

  defp duration(_event, _start), do: nil

  defp normalize_strings!(nil, _name), do: nil

  defp normalize_strings!(strings, name) when is_list(strings) do
    unless Enum.all?(strings, &is_binary/1), do: raise(ArgumentError, "#{inspect(name)} must be a list of strings")
    strings
  end

  defp normalize_strings!(other, name), do: raise(ArgumentError, "invalid #{inspect(name)}: #{inspect(other)}")

  defp normalize_count_map!(map, _name) when is_map(map) do
    unless Enum.all?(map, fn {k, v} -> is_binary(k) and is_integer(v) and v >= 0 end),
      do: raise(ArgumentError, "count map values must be non-negative integers")

    map
  end

  defp normalize_count_map!(other, name), do: raise(ArgumentError, "invalid #{name}: #{inspect(other)}")

  defp normalize_metadata!(metadata) when is_map(metadata), do: metadata

  defp normalize_metadata!(other),
    do: raise(ArgumentError, "invalid :metadata: #{inspect(other)}")
end
