defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias __MODULE__.Server
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native
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
  def global_registrar_child_specs() do
    [{Registry, keys: :unique, name: @registry}]
  end

  @doc false
  def start_worker(name, init_state) do
    case Server.start_link(name, init_state) do
      {:ok, pid} ->
        pid

      {:error, {:already_started, pid}} ->
        pid

      {:error, reason} ->
        raise reason
    end
  end

  defp handle_cb({:construct, token_ref, construct_fun, id}, init_state) do
    {:via, Registry, {@registry, id}}
    |> start_worker(init_state)
    |> GenServer.cast({:construct, token_ref, construct_fun})
  end

  @doc false
  def registry(), do: @registry

  defp normalize_run_fun(run) do
    cond do
      is_function(run, 2) ->
        run

      is_function(run, 1) ->
        fn op, _ -> run.(op) end

      true ->
        raise ArgumentError, "Invalid run function"
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

    :async =
      MLIR.CAPI.beaver_raw_create_mlir_pass(
        ctx,
        name,
        argument,
        desc,
        op,
        %{
          construct: construct,
          destruct: destruct,
          initialize: initialize,
          clone: clone,
          run: run
        }
      )

    receive do
      msg ->
        handle_cb(msg, init_state)
    end

    receive do
      {:kind, __MODULE__, ref} = msg when is_reference(ref) ->
        msg |> Beaver.Native.check!()
    end
  end
end

defmodule Beaver.MLIR.Pass.Server do
  @moduledoc false
  use GenServer
  require Logger
  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  # Client API

  def start_link(name, initial_state) do
    GenServer.start_link(__MODULE__, initial_state, name: name)
  end

  # Server Callbacks

  @impl true
  def init(initial_state) do
    {:ok, initial_state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_cast({:construct, token_ref, construct_fun}, state) do
    case invoke(token_ref, fn -> {:ok, construct_fun.(state)} end, &log_exception/1) do
      {:ok, new_state} ->
        {:noreply, new_state}

      {:exception, _kind, _reason, _stacktrace} ->
        {:noreply, nil}
    end
  end

  def handle_cast({:clone, token_ref, clone_fun, from_state}, :started_by_clone) do
    case invoke(token_ref, fn -> {:ok, clone_fun.(from_state)} end, &log_exception/1) do
      {:ok, new_state} ->
        {:noreply, {:started_by_clone, new_state}}

      {:exception, _kind, _reason, _stacktrace} ->
        {:noreply, nil}
    end
  end

  # this should only be called if it is a cloned pass
  @impl true
  def handle_info({:construct, token_ref, construct_fun, _id}, {:started_by_clone, state}) do
    handle_cast({:construct, token_ref, construct_fun}, state)
  end

  def handle_info({:initialize, token_ref, initialize_fun, _id, ctx}, state) do
    ctx = Beaver.Native.check!(ctx)

    case invoke(
           token_ref,
           fn -> initialize_fun.(ctx, state) end,
           &emit_initialize_outcome(&1, ctx)
         ) do
      {:ok, new_state} ->
        {:noreply, new_state}

      {:error, new_state} ->
        {:noreply, new_state}

      {:exception, _kind, _reason, _stacktrace} ->
        {:noreply, nil}
    end
  end

  def handle_info({:clone, token_ref, clone_fun, id, from_id}, state) do
    this = self()

    Registry.dispatch(MLIR.Pass.registry(), from_id, fn entries ->
      clone_from_owner(entries, this, id, token_ref, clone_fun, state)
    end)

    {:noreply, state}
  end

  def handle_info({:run, token_ref, run_fun, _id, op_ref}, state) do
    op = Beaver.Native.check!(op_ref)
    ctx = MLIR.context(op)

    case invoke(
           token_ref,
           fn -> {:ok, run_fun.(op, state)} end,
           &emit_run_outcome(&1, ctx)
         ) do
      {:ok, new_state} ->
        {:noreply, new_state}

      {:exception, _kind, _reason, _stacktrace} ->
        # On failure, keep the old state
        {:noreply, state}
    end
  end

  def handle_info({:destruct, token_ref, destruct_fun, _id}, state) do
    case invoke(token_ref, fn -> {:ok, destruct_fun.(state)} end, &log_exception/1) do
      {:ok, _result} ->
        {:stop, :normal, nil}

      {:exception, _kind, _reason, _stacktrace} ->
        {:stop, :normal, nil}
    end
  end

  defp invoke(token_ref, callback, before_reply) do
    CallbackRuntime.invoke(
      token_ref,
      callback,
      &MLIR.CAPI.beaver_raw_callback_reply/2,
      before_reply
    )
  end

  defp log_exception({:exception, kind, reason, stacktrace}) do
    Logger.error(Exception.format(kind, reason, stacktrace))
    Logger.flush()
  end

  defp log_exception(_outcome), do: :ok

  defp emit_initialize_outcome({:error, _state}, ctx) do
    MLIR.Location.unknown(ctx: ctx)
    |> MLIR.Diagnostic.emit("Pass initialization callback returned {:error, state}")

    Logger.flush()
  end

  defp emit_initialize_outcome({:exception, kind, reason, stacktrace}, ctx) do
    MLIR.Location.unknown(ctx: ctx)
    |> MLIR.Diagnostic.emit(Exception.format(kind, reason, stacktrace))

    Logger.flush()
  end

  defp emit_initialize_outcome(_outcome, _ctx), do: :ok

  defp emit_run_outcome({:exception, kind, reason, stacktrace}, ctx) do
    MLIR.Location.unknown(ctx: ctx)
    |> MLIR.Diagnostic.emit(Exception.format(kind, reason, stacktrace))
  end

  defp emit_run_outcome(_outcome, _ctx), do: :ok

  defp clone_from_owner(entries, owner, id, token_ref, clone_fun, state) do
    Enum.each(entries, fn {from_pid, nil} ->
      if from_pid == owner do
        {:via, Registry, {MLIR.Pass.registry(), id}}
        |> MLIR.Pass.start_worker(:started_by_clone)
        |> GenServer.cast({:clone, token_ref, clone_fun, state})
      else
        Logger.error("non-owner is requested clone")
      end
    end)
  end
end
