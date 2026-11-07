alias Beaver.MLIR.{CAPI, Diagnostic, Location}
alias Beaver.Native

defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias __MODULE__.Server
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native
  @type state() :: any()
  @callback construct(state :: state()) :: state()
  @callback run(op :: MLIR.Operation.t(), state :: state()) :: state()
  @callback initialize(ctx :: MLIR.Context.t(), state :: state()) ::
              {:ok, state()} | {:error, state()}
  @callback destruct(state :: state()) :: any()
  @callback clone(state :: state()) :: state()
  @optional_callbacks construct: 1, initialize: 2, destruct: 1, clone: 1
  require Logger

  defmacro __using__(opts) do
    quote do
      @behaviour MLIR.Pass
      Module.register_attribute(__MODULE__, :root_op, persist: true, accumulate: false)
      @root_op Keyword.get(unquote(opts), :on, "builtin.module")
      defdelegate destruct(state), to: Beaver.FallbackPass
      defdelegate initialize(ctx, state), to: Beaver.FallbackPass
      defdelegate clone(state), to: Beaver.FallbackPass
      defdelegate run(ctx, state), to: Beaver.FallbackPass
      defoverridable initialize: 2, destruct: 1, clone: 1, run: 2
    end
  end

  @registry __MODULE__.Registry
  @doc false
  def global_registrar_child_specs(), do: [{Registry, keys: :unique, name: @registry}]

  defp normalize_run_fun(run) do
    cond do
      is_function(run, 2) -> run
      is_function(run, 1) -> fn op, _ -> run.(op) end
      true -> raise ArgumentError, "Invalid run function"
    end
  end

  def create(argument, desc, op, opts) do
    name = opts[:name] || "beaver generated pass of #{argument}"
    construct = opts[:construct] || (&Beaver.FallbackPass.construct/1)
    init_state = opts[:init_state] || nil
    destruct = opts[:destruct] || (&Beaver.FallbackPass.destruct/1)
    initialize = opts[:initialize] || (&Beaver.FallbackPass.initialize/2)
    clone = opts[:clone] || (&Beaver.FallbackPass.clone/1)
    run = opts[:run] || (&Beaver.FallbackPass.run/2)
    run = normalize_run_fun(run)
    %MLIR.Context{ref: ctx} = opts[:ctx] || raise ArgumentError, "option :ctx is required"

    callbacks = %{
      construct: construct,
      destruct: destruct,
      initialize: initialize,
      clone: clone,
      run: run
    }

    {:ok, owner} = Server.start_link(callbacks, init_state)

    try do
      :async =
        CAPI.beaver_raw_create_mlir_pass(
          ctx,
          name,
          argument,
          desc,
          op,
          Map.new(callbacks, fn {key, _fun} -> {key, key} end) |> Map.put(:owner, owner)
        )

      receive do
        {:kind, __MODULE__, ref} = msg when is_reference(ref) -> Native.check!(msg)
      end
    catch
      kind, reason ->
        if Process.alive?(owner), do: GenServer.stop(owner)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end
end

defmodule Beaver.MLIR.Pass.Server do
  @moduledoc false
  use GenServer
  require Logger
  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  def start_link(callbacks, initial_state),
    do: GenServer.start_link(__MODULE__, {callbacks, initial_state})

  @impl true
  def init({callbacks, initial_state}), do: {:ok, %{callbacks: callbacks, value: initial_state}}

  @impl true
  def handle_call(:get_state, _from, state), do: {:reply, state.value, state}

  @impl true
  def handle_cast({:clone, token, id}, state) do
    Registry.register(MLIR.Pass.Registry, id, nil)
    outcome = invoke(token, fn -> {:ok, state.callbacks.clone.(state.value)} end, &log_exception/1)
    {:noreply, update_value(state, outcome)}
  end

  @impl true
  def handle_info({:construct, token, :construct, id}, state) do
    Registry.register(MLIR.Pass.Registry, id, nil)
    outcome = invoke(token, fn -> {:ok, state.callbacks.construct.(state.value)} end, &log_exception/1)
    {:noreply, update_value(state, outcome)}
  end

  def handle_info({:initialize, token, :initialize, _id, ctx}, state) do
    ctx = Native.check!(ctx)
    outcome = invoke(token, fn -> state.callbacks.initialize.(ctx, state.value) end, &emit_initialize_outcome(&1, ctx))
    {:noreply, update_value(state, outcome)}
  end

  def handle_info({:clone, token, :clone, id, _from_id}, state) do
    {:ok, clone} = start_link(state.callbacks, state.value)
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

  def handle_info({:destruct, token, :destruct, _id}, state) do
    invoke(token, fn -> {:ok, state.callbacks.destruct.(state.value)} end, &log_exception/1)
    {:stop, :normal, state}
  end

  defp update_value(state, {status, value}) when status in [:ok, :error], do: %{state | value: value}
  defp update_value(state, _exception), do: %{state | value: nil}

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
