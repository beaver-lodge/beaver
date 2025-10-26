defmodule Beaver.MLIR.RewritePatternSet do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native
  alias Beaver.MLIR
  require Logger

  defdelegate create(context), to: MLIR.CAPI, as: :mlirRewritePatternSetCreate

  @doc """
  Create a `MLIR.RewritePatternSet` from the given PDL patterns.
  The `patterns` is a list of functions that accept two arguments: `ctx` and `block`,
  and create PDL patterns in the given `block` under the given `ctx`.
  The `opts` may contain:
    * `:ctx` - (required) the `MLIR.Context` to create the patterns in.
    * `:debug` - (optional) if set to `true`, dump each created pattern for debugging.
  """
  def with_pdl_patterns(patterns, opts \\ []) do
    ctx = opts[:ctx] || raise("context is required in opts")
    pattern_module = MLIR.Location.from_env(__ENV__, ctx: ctx) |> MLIR.Module.empty()
    block = Beaver.MLIR.Module.body(pattern_module)

    for p <- patterns do
      p = p.(ctx, block)

      if opts[:debug] do
        p |> MLIR.dump!()
      end
    end

    MLIR.verify!(pattern_module)
    |> MLIR.CAPI.mlirPDLPatternModuleFromModule()
    |> MLIR.CAPI.mlirRewritePatternSetFromPDLPatternModule()
    |> then(&{&1, pattern_module})
  end

  @doc """
  Add the given `MLIR.RewritePattern` into a `MLIR.RewritePatternSet`.
  Note that the ownership of the pattern is transferred to the set after this call.
  """
  def add(set, pattern) do
    MLIR.CAPI.mlirRewritePatternSetAdd(set, pattern)
    set
  end

  @doc """
  Add a rewrite pattern defined by the given module or `match_and_rewrite` function into a `MLIR.RewritePatternSet`.
  """
  def add(set, root_name, pat, opts \\ [])

  def add(%__MODULE__{} = set, root_name, module, opts) when is_atom(module) do
    opts =
      if function_exported?(module, :construct, 1) do
        put_in(opts, [:construct], &module.construct/1)
      else
        opts
      end

    opts =
      if function_exported?(module, :destruct, 1) do
        put_in(opts, [:destruct], &module.destruct/1)
      else
        opts
      end

    opts = put_in(opts, [:match_and_rewrite], &module.match_and_rewrite/4)

    MLIR.RewritePattern.create(root_name, opts)
    |> then(&add(set, &1))
  end

  def add(%__MODULE__{} = set, root_name, match_and_rewrite, opts)
      when is_function(match_and_rewrite, 4) do
    benefit = opts[:benefit] || 1
    ctx = opts[:ctx] || raise "ctx is required in opts"

    MLIR.RewritePattern.create(root_name,
      ctx: ctx,
      benefit: benefit,
      match_and_rewrite: match_and_rewrite
    )
    |> then(&add(set, &1))
  end

  defdelegate destroy(set), to: MLIR.CAPI, as: :mlirRewritePatternSetDestroy

  defp do_destroy(%MLIR.Context{ref: ctx}, %__MODULE__{ref: set}) do
    :async = MLIR.CAPI.beaver_raw_destroy_rewrite_pattern_set(ctx, set)
  end

  defp do_destroy(%MLIR.Context{ref: ctx}, %MLIR.FrozenRewritePatternSet{ref: set}) do
    :async = MLIR.CAPI.beaver_raw_destroy_frozen_rewrite_pattern_set(ctx, set)
  end

  defp dispatch_loop(timeout \\ 2) do
    receive do
      :destroy_done ->
        :ok
    after
      timeout * 1_000 ->
        Logger.error("Timeout waiting for pattern destroy, timeout: #{inspect(timeout)}")
        Logger.flush()
        dispatch_loop(timeout * 2)
    end
  end

  @doc """
  Use thread pool of the given `ctx` to destroy the given `MLIR.RewritePatternSet` or `MLIR.FrozenRewritePatternSet`.
  """
  def threaded_destroy(ctx, set) do
    do_destroy(ctx, set)
    dispatch_loop()
  end

  @doc """
  Freeze the given `MLIR.RewritePatternSet` to a `MLIR.FrozenRewritePatternSet`.
  Note that the ownership of the input set is transferred into the frozen set after this call.
  """
  defdelegate freeze(set), to: MLIR.CAPI, as: :mlirFreezeRewritePattern
end
