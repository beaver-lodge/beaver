defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
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

  @doc """
  Ensure all passes are registered with the global registry.
  """
  def ensure_all_registered!() do
    :ok = Beaver.MLIR.CAPI.beaver_raw_register_all_passes()
  end

  @registry __MODULE__.Registry
  @doc false
  def global_registrar_child_specs() do
    [Task.child_spec(&ensure_all_registered!/0), {Registry, keys: :unique, name: @registry}]
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

  defp safe_cast(pid, token_ref, ctx, f) do
    Agent.cast(pid, fn state ->
      try do
        f.(state)
        |> tap(fn _ ->
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, true)
        end)
      rescue
        exception ->
          MLIR.Location.unknown(ctx: ctx)
          |> MLIR.Diagnostic.emit(Exception.format(:error, exception, __STACKTRACE__))

          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
          state
      catch
        {:init_err, state} ->
          MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref, false)
          state
      end
    end)
  end

  @doc false
  def handle_cb({:initialize, token_ref, initialize, id, ctx_ref}) do
    ctx = %MLIR.Context{ref: ctx_ref}

    {:via, Registry, {@registry, id, ctx}}
    |> start_worker()
    |> safe_cast(token_ref, ctx, fn state ->
      case initialize.(ctx, state) do
        {:ok, state} ->
          state

        {:error, state} ->
          throw({:init_err, state})

        _ ->
          raise ArgumentError, "Failed to initialize pass"
      end
    end)
  end

  def handle_cb({:clone, token_ref, clone, id, from_id}) do
    Registry.dispatch(@registry, from_id, fn entries ->
      for {from_pid, ctx} <- entries do
        {:via, Registry, {@registry, id, ctx}}
        |> start_worker()
        |> safe_cast(token_ref, ctx, fn nil ->
          clone.(Agent.get(from_pid, & &1))
        end)
      end
    end)
  end

  def handle_cb({:destruct, token_ref, destruct, id}) do
    Registry.dispatch(@registry, id, fn entries ->
      for {pid, ctx} <- entries do
        safe_cast(pid, token_ref, ctx, fn state ->
          :ok = destruct.(state)
        end)

        :ok = Agent.stop(pid)
      end
    end)
  end

  def handle_cb({:run, token_ref, run, id, op_ref}) do
    Registry.dispatch(@registry, id, fn entries ->
      for {pid, ctx} <- entries do
        safe_cast(pid, token_ref, ctx, fn state ->
          op = %MLIR.Operation{ref: op_ref}
          run.(op, state)
        end)
      end
    end)
  end

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
