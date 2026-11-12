alias Beaver.MLIR.CAPI
alias Beaver.Native

defmodule Beaver.MLIR.Pass do
  @moduledoc "This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}."
  alias Beaver.MLIR
  alias Beaver.MLIR.CallbackServer
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native
  @type state() :: any()
  @callback construct(state :: state()) :: state()
  @callback run(op :: MLIR.Operation.t(), state :: state()) :: state()
  @callback initialize(ctx :: MLIR.Context.t(), state :: state()) :: {:ok, state()} | {:error, state()}
  @callback destruct(state :: state()) :: any()
  @callback clone(state :: state()) :: state()
  @optional_callbacks construct: 1, initialize: 2, destruct: 1, clone: 1

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

  @doc false
  def callbacks(opts) do
    Map.new([construct: 1, destruct: 1, initialize: 2, clone: 1, run: 2], fn {key, arity} ->
      {key, opts[key] || Function.capture(Beaver.FallbackPass, key, arity)}
    end)
    |> Map.update!(:run, &normalize_run_fun/1)
  end

  def create(argument, desc, op, opts) do
    name = opts[:name] || "beaver generated pass of #{argument}"
    ctx = opts[:ctx] || raise ArgumentError, "option :ctx is required"
    callbacks = callbacks(opts)

    CallbackServer.create(__MODULE__, callbacks, opts[:init_state] || nil, fn owner ->
      Beaver.Execution.run(ctx, :create_pass, {name, argument, desc, op, owner})
    end)
  end
end
