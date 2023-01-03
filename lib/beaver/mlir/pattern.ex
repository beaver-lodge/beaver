defmodule Beaver.MLIR.Pattern do
  alias Beaver.MLIR
  import MLIR.Sigils
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Although this module is `MLIR.Pattern`, at this point it is a synonym of PDL patterns.
  Pattern-matching is done by MLIR which works in a different way from Erlang pattern-matching.
  The major difference is that MLIR pattern-matching will greedily match the patterns and maximize the benifit.
  Compiled patterns will be saved as module attributes in MLIR assembly format.
  """

  def from_string(pdl_pattern_str, opts \\ []) when is_binary(pdl_pattern_str) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        pattern_module = ~m{#{pdl_pattern_str}}.(ctx)
        if MLIR.is_null(pattern_module), do: raise("fail to parse module")
        MLIR.Operation.verify!(pattern_module)
        pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
        pdl_pattern
      end
    )
  end

  @apply_default_opts [debug: false]
  @doc """
  Apply patterns on a container (region, operation, module).
  It returns the container if it succeeds otherwise it raises.
  """
  def apply!(op, patterns, opts \\ @apply_default_opts) do
    with {:ok, module} <- apply_(op, patterns, opts) do
      module
    end
  end

  @doc """
  Apply patterns on a container (region, operation, module).
  It is named `apply_` with a underscore to avoid name collision with `Kernel.apply/2`
  """
  def apply_(op, patterns, opts \\ @apply_default_opts) when is_list(patterns) do
    if MLIR.is_null(op), do: raise("op is null")
    ctx = MLIR.Operation.from_module(op) |> MLIR.CAPI.mlirOperationGetContext()
    pattern_set = MLIR.PatternSet.get(ctx)

    for p <- patterns do
      p = p |> Beaver.Deferred.create(ctx)

      if opts[:debug] do
        p |> MLIR.dump!()
      end

      MLIR.PatternSet.insert(pattern_set, p)
    end

    MLIR.PatternSet.apply_(op, pattern_set)
  end
end
