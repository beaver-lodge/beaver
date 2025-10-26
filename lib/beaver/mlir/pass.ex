defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias __MODULE__.Server
  use Kinda.ResourceKind, forward_module: Beaver.Native
  @type state() :: any()
  @callback run(op :: MLIR.Operation.t(), state :: state()) :: state()
  @callback initialize(ctx :: MLIR.Context.t(), state :: state()) ::
              {:ok, state()} | {:error, state()}
  @callback destruct(state :: state()) :: :ok
  @callback clone(state :: state()) :: state()
  @optional_callbacks initialize: 2, destruct: 1, clone: 1
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

  @doc false
  def handle_cb({:initialize, token_ref, initialize_fun, id, ctx_ref}, init_state) do
    ctx = Beaver.Native.check!(ctx_ref)

    pid =
      {:via, Registry, {@registry, id, ctx}}
      |> start_worker(init_state)

    GenServer.cast(pid, {:initialize, token_ref, initialize_fun, ctx})
  end

  @doc false
  def registry(), do: @registry

  def create(argument, desc, op, callbacks) do
    argument_ref = MLIR.StringRef.create(argument).ref
    destruct = callbacks[:destruct] || (&Beaver.FallbackPass.destruct/1)
    initialize = callbacks[:initialize] || (&Beaver.FallbackPass.initialize/2)
    clone = callbacks[:clone] || (&Beaver.FallbackPass.clone/1)
    run = callbacks[:run] || (&Beaver.FallbackPass.run/2)

    run =
      cond do
        is_function(run, 2) ->
          run

        is_function(run, 1) ->
          fn op, _ -> run.(op) end

        true ->
          raise ArgumentError, "Invalid run function"
      end

    MLIR.CAPI.beaver_raw_create_mlir_pass(
      argument_ref,
      argument_ref,
      MLIR.StringRef.create(desc).ref,
      MLIR.StringRef.create(op).ref,
      destruct,
      initialize,
      clone,
      run
    )
    |> Beaver.Native.check!()
  end
end

defmodule Beaver.MLIR.Pass.Server do
  @moduledoc false
  use GenServer
  require Logger
  alias Beaver.MLIR

  #
  # Client API
  #

  @doc false
  def start_link(name, initial_state) do
    GenServer.start_link(__MODULE__, initial_state, name: name)
  end

  #
  # Server Callbacks
  #

  @impl true
  def init(initial_state) do
    {:ok, initial_state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_cast({:initialize, token_ref, initialize_fun, ctx}, nil) do
    try do
      case initialize_fun.(ctx, nil) do
        {:ok, new_state} ->
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
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
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:noreply, nil}
    catch
      {:init_err, state} ->
        MLIR.Location.unknown(ctx: ctx)
        |> MLIR.Diagnostic.emit("Pass initialization callback returned {:error, state}")

        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:noreply, state}
    end
  end

  @impl true
  def handle_cast({:clone, token_ref, clone_fun, from_state, ctx}, :expect_to_initialize_by_clone) do
    try do
      new_state = clone_fun.(from_state)
      MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
      {:noreply, new_state}
    rescue
      exception ->
        MLIR.Location.unknown(ctx: ctx)
        |> MLIR.Diagnostic.emit(Exception.format(:error, exception, __STACKTRACE__))

        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:noreply, nil}
    end
  end

  @impl true
  def handle_info({:clone, token_ref, clone_fun, id, from_id}, state) do
    Registry.dispatch(MLIR.Pass.registry(), from_id, fn entries ->
      for {from_pid, ctx} <- entries do
        if from_pid != self() do
          raise "Should start clone from the original state owner"
        end

        pid =
          {:via, Registry, {MLIR.Pass.registry(), id, ctx}}
          |> MLIR.Pass.start_worker(:expect_to_initialize_by_clone)

        GenServer.cast(pid, {:clone, token_ref, clone_fun, state, ctx})
      end
    end)

    {:noreply, state}
  end

  def handle_info({:run, token_ref, run_fun, _id, op_ref}, state) do
    op = Beaver.Native.check!(op_ref)
    ctx = MLIR.context(op)

    try do
      # The `run` callback returns the new state
      new_state = run_fun.(op, state)
      MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
      {:noreply, new_state}
    rescue
      exception ->
        MLIR.Location.unknown(ctx: ctx)
        |> MLIR.Diagnostic.emit(Exception.format(:error, exception, __STACKTRACE__))

        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        # On failure, keep the old state
        {:noreply, state}
    end
  end

  def handle_info({:destruct, token_ref, destruct_fun, _id}, state) do
    try do
      :ok = destruct_fun.(state)
      MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
      {:stop, :normal, state}
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        # Stop the server even on failure
        {:stop, :normal, state}
    end
  end
end
