defmodule Beaver.MLIR.RewritePatternSet do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native
  alias Beaver.MLIR
  require Logger

  defdelegate create(context), to: MLIR.CAPI, as: :mlirRewritePatternSetCreate

  @doc """
  Add the given `MLIR.RewritePattern` into a `MLIR.RewritePatternSet`.
  Note that the ownership of the pattern is transferred to the set after this call.
  """
  defdelegate add(set, pattern), to: MLIR.CAPI, as: :mlirRewritePatternSetAdd

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

  defp dispatch_loop() do
    receive do
      :destroy_done ->
        :ok

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

  def destroy(ctx, set) do
    do_destroy(ctx, set)
    dispatch_loop()
  end

  @doc """
  Freeze the given `MLIR.RewritePatternSet` to a `MLIR.FrozenRewritePatternSet`.
  Note that the ownership of the input set is transferred into the frozen set after this call.
  """
  defdelegate freeze(set), to: MLIR.CAPI, as: :mlirFreezeRewritePattern
end
