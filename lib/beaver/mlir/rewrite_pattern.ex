alias Beaver.MLIR.CAPI
alias Beaver.Native

defmodule Beaver.MLIR.RewritePattern do
  @moduledoc """
  This module defines functions working with MLIR Rewrite Patterns.
  """
  require Logger
  alias Beaver.MLIR
  alias __MODULE__.Server
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native, fields: [owner: nil]
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
  def global_registrar_child_specs(), do: [{Registry, keys: :unique, name: @registry}]

  def create(root_name, opts) do
    init_state = opts[:init_state]
    benefit = opts[:benefit] || 1
    ctx = opts[:ctx] || raise ArgumentError, "option :ctx is required"
    construct = opts[:construct] || (&Beaver.FallbackPattern.construct/1)
    destruct = opts[:destruct] || (&Beaver.FallbackPattern.destruct/1)

    match_and_rewrite = opts[:match_and_rewrite] || (&Beaver.FallbackPattern.match_and_rewrite/4)

    callbacks = %{construct: construct, destruct: destruct, matchAndRewrite: match_and_rewrite}
    {:ok, owner} = Server.start_link(callbacks, init_state)

    try do
      :async =
        CAPI.beaver_raw_create_mlir_rewrite_pattern(
          root_name,
          benefit,
          ctx.ref,
          owner,
          Keyword.get(opts, :prefilter, {-1, -1, []})
        )

      receive do
        {:kind, __MODULE__, ref} = msg when is_reference(ref) -> %{Native.check!(msg) | owner: owner}
      end
    catch
      kind, reason ->
        if Process.alive?(owner), do: GenServer.stop(owner)
        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  @doc "Number of BEAM match callbacks serviced by this live pattern."
  def callback_count(%__MODULE__{owner: owner}) when is_pid(owner), do: GenServer.call(owner, :callback_count)
end

defmodule Beaver.MLIR.RewritePattern.Server do
  @moduledoc false
  use GenServer
  require Logger
  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  def start_link(callbacks, value), do: GenServer.start_link(__MODULE__, {callbacks, value})

  @impl true
  def init({callbacks, value}), do: {:ok, %{callbacks: callbacks, value: value, count: 0}}

  @impl true
  def handle_call(:callback_count, _from, state), do: {:reply, state.count, state}

  @impl true
  def handle_info({:construct, token, :construct, id}, state) do
    Registry.register(MLIR.RewritePattern.Registry, id, nil)
    outcome = invoke(token, fn -> {:ok, state.callbacks.construct.(state.value)} end, &log_exception/1)
    {:noreply, update_value(state, outcome)}
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
  defp update_value(state, _exception), do: state

  defp invoke(token_ref, callback, before_reply),
    do: CallbackRuntime.invoke(token_ref, callback, &CAPI.beaver_raw_callback_reply/2, before_reply)

  defp log_exception({:exception, kind, reason, stacktrace}) do
    Logger.error(Exception.format(kind, reason, stacktrace))
    Logger.flush()
  end

  defp log_exception(_outcome), do: :ok
end
