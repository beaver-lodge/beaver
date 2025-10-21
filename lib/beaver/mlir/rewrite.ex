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

  defp dispatch_loop() do
    receive do
      {{:kind, MLIR.LogicalResult, _} = ret, diagnostics} ->
        {Beaver.Native.check!(ret), diagnostics}

      msg ->
        try do
          :ok = MLIR.RewritePattern.handle_cb(msg)
        rescue
          exception ->
            Logger.error(Exception.format(:error, exception, __STACKTRACE__))
            Logger.flush()
        end

        dispatch_loop()
    end
  end

  def apply_patterns(ir, pattern_set) do
    cfg = MLIR.CAPI.beaverGreedyRewriteDriverConfigGet()
    ctx = MLIR.context(ir)
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
      MLIR.FrozenRewritePatternSet.destroy(ctx, frozen_set)
    end
  end
end
