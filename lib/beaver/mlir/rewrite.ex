defmodule Beaver.MLIR.Rewrite do
  @moduledoc """
  This module defines functions working with MLIR rewrite patterns and pattern sets.
  """
  alias Beaver.MLIR
  require Logger

  defp do_apply(
         %MLIR.Context{ref: ctx},
         %MLIR.Module{ref: mod},
         %MLIR.FrozenRewritePatternSet{ref: set},
         cfg
       ) do
    MLIR.CAPI.beaver_raw_apply_rewrite_pattern_set_with_module(ctx, mod, set, cfg)
  end

  defp do_apply(
         %MLIR.Context{ref: ctx},
         %MLIR.Operation{ref: op},
         %MLIR.FrozenRewritePatternSet{ref: set},
         cfg
       ) do
    MLIR.CAPI.beaver_raw_apply_rewrite_pattern_set_with_op(ctx, op, set, cfg)
  end

  defp dispatch_loop(timeout \\ 2) do
    receive do
      {{:kind, MLIR.LogicalResult, _} = ret, diagnostics} ->
        {Beaver.Native.check!(ret), diagnostics}
    after
      timeout * 1_000 ->
        Logger.error("Timeout waiting for pattern set application, timeout: #{inspect(timeout)}s")
        Logger.flush()
        dispatch_loop(timeout * 2)
    end
  end

  @ctx_for_pat_apply __MODULE__.CtxForPatApply
  @doc false
  def thread_pool_child_spec() do
    spec = Agent.child_spec(fn -> MLIR.Context.create() end)

    update_in(spec.start, fn {Agent, :start_link, [fun]} ->
      opts = [name: @ctx_for_pat_apply]
      {Agent, :start_link, [fun, opts]}
    end)
  end

  def stop_thread_pool() do
    Agent.get(@ctx_for_pat_apply, &MLIR.Context.destroy/1)
    Agent.stop(@ctx_for_pat_apply)
  end

  def apply_patterns(ir, pattern_set) do
    cfg = MLIR.CAPI.beaverGreedyRewriteDriverConfigGet()
    ctx = Agent.get(@ctx_for_pat_apply, & &1)
    :async = do_apply(ctx, ir, pattern_set, cfg)
    dispatch_loop()
  end

  def apply_patterns!(ir, %MLIR.FrozenRewritePatternSet{} = pattern_set) do
    {res, diagnostics} = apply_patterns(ir, pattern_set)

    unless Enum.empty?(diagnostics) do
      Logger.error(MLIR.Diagnostic.format(diagnostics, "failed to apply pattern set."))
    end

    Logger.flush()

    if MLIR.LogicalResult.success?(res) do
      ir
    else
      raise "pattern application failed to converge"
    end
  end

  def apply_patterns!(ir, patterns) when is_list(patterns) do
    ctx = MLIR.context(ir)
    set = MLIR.RewritePatternSet.create(ctx)

    for {root, match_and_rewrite} <- patterns do
      MLIR.RewritePatternSet.add(set, root, match_and_rewrite, ctx: ctx)
    end

    frozen_set = %MLIR.FrozenRewritePatternSet{} = MLIR.RewritePatternSet.freeze(set)

    try do
      apply_patterns!(ir, frozen_set)
    after
      MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen_set)
    end
  end
end
