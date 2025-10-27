defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias __MODULE__.Server
  use Kinda.ResourceKind, forward_module: Beaver.Native
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
    argument_ref = MLIR.StringRef.create(argument).ref
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
        argument_ref,
        argument_ref,
        MLIR.StringRef.create(desc).ref,
        MLIR.StringRef.create(op).ref,
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
    try do
      new_state = construct_fun.(state)
      MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, true)
      {:noreply, new_state}
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, false)
        {:noreply, nil}
    end
  end

  def handle_cast({:clone, token_ref, clone_fun, from_state}, :started_by_clone) do
    try do
      new_state = clone_fun.(from_state)
      MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, true)
      {:noreply, {:started_by_clone, new_state}}
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, false)
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

    try do
      case initialize_fun.(ctx, state) do
        {:ok, new_state} ->
          MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, true)
          {:noreply, new_state}

        {:error, state} ->
          # This throw is caught by the `catch` block below
          throw({:init_err, state})

        _ ->
          raise ArgumentError, "Failed to initialize pass"
      end
    rescue
      exception ->
        MLIR.Location.unknown(ctx: ctx)
        |> MLIR.Diagnostic.emit(Exception.format(:error, exception, __STACKTRACE__))

        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, false)
        {:noreply, nil}
    catch
      {:init_err, state} ->
        MLIR.Location.unknown(ctx: ctx)
        |> MLIR.Diagnostic.emit("Pass initialization callback returned {:error, state}")

        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, false)
        {:noreply, state}
    end
  end

  def handle_info({:clone, token_ref, clone_fun, id, from_id}, state) do
    this = self()

    Registry.dispatch(MLIR.Pass.registry(), from_id, fn entries ->
      for from <- entries do
        {from_pid, nil} = from

        if from_pid != this do
          raise "Should start clone from the original state owner"
        end

        {:via, Registry, {MLIR.Pass.registry(), id}}
        |> MLIR.Pass.start_worker(:started_by_clone)
        |> GenServer.cast({:clone, token_ref, clone_fun, state})
      end
    end)

    {:noreply, state}
  end

  def handle_info({:run, token_ref, run_fun, _id, op_ref}, state) do
    op = Beaver.Native.check!(op_ref)
    ctx = MLIR.context(op)

    try do
      new_state = run_fun.(op, state)
      MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, true)
      {:noreply, new_state}
    rescue
      exception ->
        MLIR.Location.unknown(ctx: ctx)
        |> MLIR.Diagnostic.emit(Exception.format(:error, exception, __STACKTRACE__))

        MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, false)
        # On failure, keep the old state
        {:noreply, state}
    end
  end

  def handle_info({:destruct, token_ref, destruct_fun, _id}, state) do
    try do
      destruct_fun.(state)
      MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, true)
      {:stop, :normal, nil}
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_signal(token_ref, false)
        {:stop, :normal, nil}
    end
  end
end
