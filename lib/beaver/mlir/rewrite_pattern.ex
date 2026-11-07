alias Beaver.MLIR.CAPI
alias Beaver.Native

defmodule Beaver.MLIR.RewritePattern do
  @moduledoc """
  This module defines functions working with MLIR Rewrite Patterns.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.CallbackServer
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

    CallbackServer.create(__MODULE__, callbacks, init_state, fn owner ->
      pattern =
        Beaver.Execution.run(
          ctx,
          :create_pattern,
          {root_name, benefit, owner, Keyword.get(opts, :prefilter, {-1, -1, []})}
        )

      %{pattern | owner: owner}
    end)
  end

  @doc "Number of BEAM match callbacks serviced by this live pattern."
  def callback_count(%__MODULE__{owner: owner}) when is_pid(owner), do: GenServer.call(owner, :callback_count)
end
