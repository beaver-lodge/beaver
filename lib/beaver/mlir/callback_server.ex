defmodule Beaver.MLIR.CallbackServer do
  @moduledoc false
  use GenServer
  require Logger
  alias Beaver.{MLIR, Native}
  alias Beaver.MLIR.{CAPI, Diagnostic, Location}
  alias Kinda.CallbackRuntime

  def start_link(kind, callbacks, value),
    do: GenServer.start_link(__MODULE__, %{kind: kind, callbacks: callbacks, value: value, count: 0})

  def create(kind, callbacks, value, create) do
    {:ok, owner} = start_link(kind, callbacks, value)

    try do
      create.(owner)
    catch
      kind, reason ->
        if Process.alive?(owner), do: GenServer.stop(owner)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  @impl true
  def init(state), do: {:ok, state}

  @impl true
  def handle_call(:get_state, _from, state), do: {:reply, state.value, state}
  def handle_call(:callback_count, _from, state), do: {:reply, state.count, state}

  @impl true
  def handle_cast({:clone, token, id}, state) do
    Registry.register(Module.concat(state.kind, Registry), id, nil)
    outcome = invoke(token, fn -> {:ok, state.callbacks.clone.(state.value)} end, &log_exception/1)
    {:noreply, update_value(state, outcome)}
  end

  @impl true
  def handle_info({:construct, token, :construct, id}, state) do
    Registry.register(Module.concat(state.kind, Registry), id, nil)
    outcome = invoke(token, fn -> {:ok, state.callbacks.construct.(state.value)} end, &log_exception/1)
    {:noreply, update_value(state, outcome)}
  end

  def handle_info({:initialize, token, :initialize, _id, ctx}, state) do
    ctx = Native.check!(ctx)
    outcome = invoke(token, fn -> state.callbacks.initialize.(ctx, state.value) end, &emit_initialize_outcome(&1, ctx))
    {:noreply, update_value(state, outcome)}
  end

  def handle_info({:clone, token, :clone, id, _from_id}, state) do
    {:ok, clone} = start_link(state.kind, state.callbacks, state.value)
    GenServer.cast(clone, {:clone, token, id})
    {:noreply, state}
  end

  def handle_info({:run, token, :run, _id, op_ref}, state) do
    op = Native.check!(op_ref)
    ctx = MLIR.context(op)
    outcome = invoke(token, fn -> {:ok, state.callbacks.run.(op, state.value)} end, &emit_run_outcome(&1, ctx))

    case outcome do
      {:ok, value} -> {:noreply, %{state | value: value}}
      _ -> {:noreply, state}
    end
  end

  def handle_info({:matchAndRewrite, token, :matchAndRewrite, _id, pattern, op, rewriter}, state) do
    outcome =
      invoke(
        token,
        fn ->
          [pattern, op, rewriter] = Enum.map([pattern, op, rewriter], &Native.check!/1)
          state.callbacks.matchAndRewrite.(pattern, op, rewriter, state.value)
        end,
        &log_exception/1
      )

    {:noreply, update_value(%{state | count: state.count + 1}, outcome)}
  end

  def handle_info({:destruct, token, :destruct, _id}, state) do
    invoke(token, fn -> {:ok, state.callbacks.destruct.(state.value)} end, &log_exception/1)
    {:stop, :normal, state}
  end

  defp update_value(state, {status, value}) when status in [:ok, :error], do: %{state | value: value}
  defp update_value(%{kind: MLIR.Pass} = state, _exception), do: %{state | value: nil}
  defp update_value(state, _exception), do: state

  defp invoke(token_ref, callback, before_reply),
    do: CallbackRuntime.invoke(token_ref, callback, &CAPI.beaver_raw_callback_reply/2, before_reply)

  defp log_exception({:exception, kind, reason, stacktrace}) do
    Logger.error(Exception.format(kind, reason, stacktrace))
    Logger.flush()
  end

  defp log_exception(_outcome), do: :ok

  defp emit_initialize_outcome({:error, _state}, ctx) do
    Location.unknown(ctx: ctx) |> Diagnostic.emit("Pass initialization callback returned {:error, state}")
    Logger.flush()
  end

  defp emit_initialize_outcome({:exception, kind, reason, stacktrace}, ctx) do
    Location.unknown(ctx: ctx) |> Diagnostic.emit(Exception.format(kind, reason, stacktrace))
    Logger.flush()
  end

  defp emit_initialize_outcome(_outcome, _ctx), do: :ok

  defp emit_run_outcome({:exception, kind, reason, stacktrace}, ctx),
    do: Location.unknown(ctx: ctx) |> Diagnostic.emit(Exception.format(kind, reason, stacktrace))

  defp emit_run_outcome(_outcome, _ctx), do: :ok
end
