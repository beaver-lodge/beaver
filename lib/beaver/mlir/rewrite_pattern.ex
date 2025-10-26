defmodule Beaver.MLIR.RewritePattern do
  @moduledoc """
  This module defines functions working with MLIR Rewrite Patterns.
  """
  require Logger
  alias Beaver.MLIR
  alias __MODULE__.Server
  use Kinda.ResourceKind, forward_module: Beaver.Native
  @type state() :: any()

  @callback construct(state :: state()) :: state()
  @callback destruct(state :: state()) :: any()
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

  defp start_worker(name, init_state) do
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
  def handle_cb({:construct, token_ref, construct, id}, init_state) do
    pid =
      {:via, Registry, {@registry, id}}
      |> start_worker(init_state)

    # Cast to the GenServer worker to handle construction
    GenServer.cast(pid, {:construct, token_ref, construct})
  end

  def create(root_name, opts) do
    init_state = opts[:init_state]
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
      msg -> handle_cb(msg, init_state)
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

  def start_link(name, state) do
    GenServer.start_link(__MODULE__, state, name: name)
  end

  # Server Callbacks

  @impl true
  def init(initial_state) do
    {:ok, initial_state}
  end

  @impl true
  def handle_cast({:construct, token_ref, construct_fun}, state) do
    try do
      new_state = construct_fun.(state)
      MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
      {:noreply, new_state}
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(
        {:matchAndRewrite, token_ref, match_and_rewrite, _id, pattern, op, rewriter},
        state
      ) do
    try do
      kind_args = Enum.map([pattern, op, rewriter], &Beaver.Native.check!/1)

      case apply(match_and_rewrite, kind_args ++ [state]) do
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
  def handle_info({:destruct = _cb, token_ref, destruct, _id}, state) do
    try do
      destruct.(state)
      MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
      {:stop, :normal, nil}
    rescue
      exception ->
        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
        Logger.flush()
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
        {:stop, :normal, nil}
    end
  end
end
