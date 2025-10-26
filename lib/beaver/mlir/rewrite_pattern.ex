defmodule Beaver.MLIR.RewritePattern do
  @moduledoc """
  This module defines functions working with MLIR Rewrite Patterns.
  """
  require Logger
  alias Beaver.MLIR
  # Alias the new GenServer worker
  alias Beaver.MLIR.RewritePattern.Server
  use Kinda.ResourceKind, forward_module: Beaver.Native
  @type state() :: any()

  @callback construct(state :: state()) :: {:ok, state()} | {:error, state()}
  @callback destruct(state :: state()) :: :ok
  @callback match_and_rewrite(
              pattern :: MLIR.RewritePattern.t(),
              op :: MLIR.Operation.t(),
              rewriter :: MLIR.PatternRewriter.t(),
              state :: state()
            ) :: {:ok, state()} | {:error, state()}

  @optional_callbacks construct: 1, destruct: 1

  defmacro __using__(_opts) do
    quote do
      @behaviour Beaver.MLIR.RewritePattern
      defdelegate construct(state), to: Beaver.FallbackPattern
      defdelegate destruct(state), to: Beaver.FallbackPattern
      defoverridable construct: 1, destruct: 1
    end
  end

  @registry __MODULE__.Registry
  @doc false
  def global_registrar_child_specs() do
    [{Registry, keys: :unique, name: @registry}]
  end

  # Updated to start the GenServer
  defp start_worker(name) do
    case Server.start_link(name) do
      {:ok, pid} ->
        pid

      {:error, {:already_started, pid}} ->
        pid

      {:error, reason} ->
        raise reason
    end
  end

  # The safe_cast/3 function is no longer needed.

  @doc false
  def handle_cb({:construct, token_ref, construct, id}) do
    pid =
      {:via, Registry, {@registry, id}}
      |> start_worker()

    # Cast to the GenServer worker to handle construction
    GenServer.cast(pid, {:construct, token_ref, construct})
  end

  def handle_cb({:destruct = _cb, token_ref, destruct, id}) do
    Registry.dispatch(@registry, id, fn entries ->
      for {pid, _ctx} <- entries do
        # Cast to the GenServer worker to handle destruction
        GenServer.cast(pid, {:destruct, token_ref, destruct})
      end
    end)
  end

  def handle_cb({:matchAndRewrite, token_ref, match_and_rewrite, id, pattern, op, rewriter}) do
    Registry.dispatch(@registry, id, fn entries ->
      for {pid, _ctx} <- entries do
        kind_args = Enum.map([pattern, op, rewriter], &Beaver.Native.check!/1)

        # Cast to the GenServer worker to handle the rewrite logic
        GenServer.cast(pid, {:match_and_rewrite, token_ref, match_and_rewrite, kind_args})
      end
    end)
  end

  def create(root_name, opts) do
    benefit = opts[:benefit] || 1
    ctx = opts[:ctx] || raise ArgumentError, "option :ctx is required"
    construct = opts[:construct] || (&Beaver.FallbackPattern.construct/1)
    destruct = opts[:destruct] || (&Beaver.FallbackPattern.destruct/1)

    match_and_rewrite =
      opts[:match_and_rewrite] || (&Beaver.FallbackPattern.match_and_rewrite/4)

    :async =
      MLIR.CAPI.beaver_raw_create_mlir_rewrite_pattern(
        root_name,
        benefit,
        ctx.ref,
        construct,
        destruct,
        match_and_rewrite
      )

    receive do
      msg -> __MODULE__.handle_cb(msg)
    end

    receive do
      {:kind, __MODULE__, ref} = msg when is_reference(ref) ->
        msg |> Beaver.Native.check!()
    end
  end
end

defmodule Beaver.MLIR.RewritePattern.Server do
  @moduledoc false
  use GenServer
  require Logger
  alias Beaver.MLIR

  # Client API

  @spec start_link(GenServer.name()) :: GenServer.on_start()
  def start_link(name) do
    # Start with nil state, just as the Agent did
    GenServer.start_link(__MODULE__, nil, name: name)
  end

  # Server Callbacks

  @impl true
  def init(initial_state) do
    {:ok, initial_state}
  end

  @impl true
  def handle_cast({:construct, token_ref, construct_fun}, state) do
    try do
      case construct_fun.(state) do
        {:ok, new_state} ->
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
          {:noreply, new_state}

        _ ->
          # This will be caught by the rescue block
          raise ArgumentError, "Failed to initialize rewrite pattern"
      end
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:noreply, state}
    end
  end

  @impl true
  def handle_cast({:destruct, token_ref, destruct_fun}, state) do
    try do
      :ok = destruct_fun.(state)
      MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
      # State remains unchanged after destruction.
      {:noreply, state}
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:noreply, state}
    end
  end

  @impl true
  def handle_cast({:match_and_rewrite, token_ref, m_and_r_fun, args}, state) do
    try do
      case apply(m_and_r_fun, args ++ [state]) do
        {:ok, new_state} ->
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
          {:noreply, new_state}

        {:error, new_state} ->
          # This throw is caught by the `catch` block below
          throw({:rewrite_err, new_state})
      end
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:noreply, state}
    catch
      {:rewrite_err, new_state} ->
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        # Store the new state, even on rewrite error
        {:noreply, new_state}
    end
  end

  @impl true
  def handle_info(msg, state) do
    Beaver.MLIR.RewritePattern.handle_cb(msg)
    {:noreply, state}
  end
end
