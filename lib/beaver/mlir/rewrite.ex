alias Beaver.MLIR.{CAPI, Context, LogicalResult, Operation, RewritePatternSet}

defmodule Beaver.MLIR.Rewrite do
  @moduledoc """
  This module defines functions working with MLIR rewrite patterns and pattern sets.
  """
  alias Beaver.MLIR
  require Logger

  @typedoc "IR surfaces accepted by the rewrite driver."
  @type rewrite_ir() :: MLIR.Module.t() | Operation.t()

  @typedoc "Processed MLIR diagnostic tree emitted during rewrite application."
  @type diagnostic() ::
          {severity :: atom(), location :: String.t(), message :: String.t(), nested :: [diagnostic()]}

  @type diagnostics() :: [diagnostic()]

  @typedoc "Opaque greedy rewrite driver config handle passed to configuration callbacks."
  @type config_handle() :: term()

  @typedoc "Callback used to customize a greedy rewrite config before application."
  @type config_callback() :: (config_handle() -> any())

  @typedoc "Supported high-level rewrite config options."
  @type config_option() ::
          {:max_iterations, integer()}
          | {:max_num_rewrites, integer()}
          | {:use_top_down_traversal, boolean()}
          | {:enable_folding, boolean()}
          | {:enable_constant_cse, boolean()}

  @type config_opts() :: [config_option()]
  @type config_input() :: config_opts() | config_callback() | nil

  @typedoc "Pattern callback entries accepted by the list-based rewrite surface."
  @type match_and_rewrite() ::
          (MLIR.RewritePattern.t(), Operation.t(), MLIR.PatternRewriter.t(), any() -> {:ok, any()} | {:error, any()})

  @type pattern_entry() :: {root_name :: term(), match_and_rewrite()}

  @type pattern_list() :: [pattern_entry()]
  @type apply_result() :: {LogicalResult.t(), diagnostics()}

  defp do_apply(%Context{ref: ctx}, %MLIR.Module{ref: mod}, %MLIR.FrozenRewritePatternSet{ref: set}, cfg),
    do: CAPI.beaver_raw_apply_rewrite_pattern_set_with_module(ctx, mod, set, cfg)

  defp do_apply(%Context{ref: ctx}, %Operation{ref: op}, %MLIR.FrozenRewritePatternSet{ref: set}, cfg),
    do: CAPI.beaver_raw_apply_rewrite_pattern_set_with_op(ctx, op, set, cfg)

  defp dispatch_loop(timeout \\ 2) do
    receive do
      {{:kind, LogicalResult, _} = ret, diagnostics} -> {Beaver.Native.check!(ret), diagnostics}
    after
      timeout * 1_000 ->
        Logger.error("Timeout waiting for pattern set application, timeout: #{inspect(timeout)}s")
        Logger.flush()
        dispatch_loop(timeout * 2)
    end
  end

  @ctx_for_pat_apply __MODULE__.CtxForPatApply
  @doc false
  @spec thread_pool_child_spec() :: Supervisor.child_spec()
  def thread_pool_child_spec() do
    spec = Agent.child_spec(fn -> Context.create() end)

    update_in(spec.start, fn {Agent, :start_link, [fun]} ->
      opts = [name: @ctx_for_pat_apply]
      {Agent, :start_link, [fun, opts]}
    end)
  end

  @spec stop_thread_pool() :: :ok
  def stop_thread_pool() do
    Agent.get(@ctx_for_pat_apply, &Context.destroy/1)
    Agent.stop(@ctx_for_pat_apply)
  end

  @spec with_default_config((config_handle() -> result)) :: result when result: var
  def with_default_config(fun) when is_function(fun, 1),
    do:
      Beaver.Native.with_resource(
        CAPI.mlirGreedyRewriteDriverConfigCreate(),
        &CAPI.mlirGreedyRewriteDriverConfigDestroy/1,
        fun
      )

  @spec with_default_config(config_input(), (config_handle() -> result)) :: result
        when result: var
  def with_default_config(configure, fun)
      when (is_nil(configure) or is_list(configure) or is_function(configure, 1)) and
             is_function(fun, 1) do
    configure = to_config_callback(configure)

    with_default_config(fn cfg ->
      _ = configure.(cfg)
      fun.(cfg)
    end)
  end

  defp noop_config_callback(_cfg), do: :ok

  defp create_pattern_set(ctx, patterns) do
    set = RewritePatternSet.create(ctx)

    Enum.each(patterns, fn
      {root, match_and_rewrite} -> RewritePatternSet.add(set, root, match_and_rewrite, ctx: ctx)
    end)

    set
  end

  @spec to_config_callback(config_input()) :: config_callback()
  def to_config_callback(nil), do: &noop_config_callback/1
  def to_config_callback(callback) when is_function(callback, 1), do: callback

  def to_config_callback(opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "rewrite config opts must be a keyword list")

    fn cfg ->
      Enum.each(opts, fn
        {:max_iterations, value} -> CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, value)
        {:max_num_rewrites, value} -> CAPI.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(cfg, value)
        {:use_top_down_traversal, value} -> CAPI.mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(cfg, value)
        {:enable_folding, value} -> CAPI.mlirGreedyRewriteDriverConfigEnableFolding(cfg, value)
        {:enable_constant_cse, value} -> CAPI.mlirGreedyRewriteDriverConfigEnableConstantCSE(cfg, value)
        {key, _value} -> raise ArgumentError, "unsupported rewrite config option: #{inspect(key)}"
      end)
    end
  end

  @type pattern_input() :: pattern_list() | RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t()

  @spec apply_patterns(rewrite_ir(), pattern_input()) :: apply_result()
  @spec apply_patterns(rewrite_ir(), pattern_input(), config_input()) :: apply_result()
  def apply_patterns(ir, patterns, configure \\ nil)

  def apply_patterns(ir, patterns, configure) when is_nil(configure) or is_list(configure),
    do: apply_patterns(ir, patterns, to_config_callback(configure))

  def apply_patterns(ir, %MLIR.FrozenRewritePatternSet{} = patterns, configure)
      when is_function(configure, 1) do
    ctx = Agent.get(@ctx_for_pat_apply, & &1)

    with_default_config(fn cfg ->
      _ = configure.(cfg)
      :async = do_apply(ctx, ir, patterns, cfg)
      dispatch_loop()
    end)
  end

  def apply_patterns(ir, patterns, configure)
      when (is_list(patterns) or is_struct(patterns, RewritePatternSet)) and
             is_function(configure, 1) do
    ctx = MLIR.context(ir)
    set = if is_list(patterns), do: create_pattern_set(ctx, patterns), else: patterns
    frozen = %MLIR.FrozenRewritePatternSet{} = RewritePatternSet.freeze(set)

    try do
      apply_patterns(ir, frozen, configure)
    after
      MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen)
    end
  end

  @spec apply_patterns!(rewrite_ir(), pattern_input()) :: rewrite_ir()
  @spec apply_patterns!(rewrite_ir(), pattern_input(), config_input()) :: rewrite_ir()
  def apply_patterns!(ir, patterns, configure \\ nil) do
    {result, diagnostics} = apply_patterns(ir, patterns, configure)

    unless Enum.empty?(diagnostics),
      do: Logger.error(MLIR.Diagnostic.format(diagnostics, "failed to apply pattern set."))

    Logger.flush()

    if LogicalResult.success?(result),
      do: ir,
      else: raise("pattern application failed to converge")
  end
end
