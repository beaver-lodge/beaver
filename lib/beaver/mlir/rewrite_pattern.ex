defmodule Beaver.MLIR.RewritePattern do
  @moduledoc """
  This module defines functions working with MLIR Rewrite Patterns.
  """
  require Logger
  alias Beaver.MLIR
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

  defp start_worker(name) do
    case Agent.start_link(fn -> nil end, name: name) do
      {:ok, pid} ->
        pid

      {:error, {:already_started, pid}} ->
        pid

      {:error, reason} ->
        raise reason
    end
  end

  defp safe_cast(pid, token_ref, f) do
    Agent.cast(pid, fn state ->
      try do
        f.(state)
        |> tap(fn _ ->
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
        end)
      rescue
        exception ->
          Logger.error(Exception.format(:error, exception, __STACKTRACE__))
          Logger.flush()
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
          state
      catch
        {:rewrite_err, state} ->
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
          state
      end
    end)
  end

  @doc false
  def handle_cb({:construct, token_ref, construct, id}) do
    {:via, Registry, {@registry, id}}
    |> start_worker()
    |> safe_cast(token_ref, fn state ->
      case construct.(state) do
        {:ok, state} ->
          state

        _ ->
          raise ArgumentError, "Failed to initialize rewrite pattern"
      end
    end)
  end

  def handle_cb({:destruct, token_ref, destruct, id}) do
    Registry.dispatch(@registry, id, fn entries ->
      for {pid, _ctx} <- entries do
        safe_cast(pid, token_ref, fn state ->
          :ok = destruct.(state)
        end)

        :ok = Agent.stop(pid)
      end
    end)
  end

  def handle_cb({:matchAndRewrite, token_ref, match_and_rewrite, id, pattern, op, rewriter}) do
    Registry.dispatch(@registry, id, fn entries ->
      for {pid, _ctx} <- entries do
        kind_args = Enum.map([pattern, op, rewriter], &Beaver.Native.check!/1)

        safe_cast(
          pid,
          token_ref,
          &case apply(match_and_rewrite, kind_args ++ [&1]) do
            {:ok, state} ->
              state

            {:error, state} ->
              throw({:rewrite_err, state})
          end
        )
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
