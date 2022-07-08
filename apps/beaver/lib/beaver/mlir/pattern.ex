defmodule Beaver.MLIR.Pattern do
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Although this module is `MLIR.Pattern`, at this point it is a synonym of PDL patterns.
  Pattern-matching is done by MLIR which works in a different way from Erlang pattern-matching.
  The major difference is that MLIR pattern-matching will greedily match the patterns and maximize the benifit.
  Compiled patterns will be saved as module attributes in MLIR assembly format.
  """

  def from_string(pdl_pattern_str) when is_binary(pdl_pattern_str) do
    pattern_module = ~m{#{pdl_pattern_str}}
    if MLIR.Module.is_null(pattern_module), do: raise("fail to parse module")
    MLIR.Operation.verify!(pattern_module)
    pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
    pdl_pattern
  end

  @doc """
  Apply patterns on a container (region, operation, module).
  It returns the container if it succeeds otherwise it raises.
  """
  def apply!(op, patterns) when is_list(patterns) do
    pattern_set = MLIR.PatternSet.get()

    for p <- patterns do
      MLIR.PatternSet.insert(pattern_set, p)
    end

    MLIR.PatternSet.apply!(op, pattern_set)
  end

  @doc """
  Apply patterns on a container (region, operation, module).
  It is named `apply_` with a underscore to avoid name collision with `Kernel.apply/2`
  """
  def apply_(op, patterns) when is_list(patterns) do
    pattern_set = MLIR.PatternSet.get()

    for p <- patterns do
      MLIR.PatternSet.insert(pattern_set, p)
    end

    MLIR.PatternSet.apply_(op, pattern_set)
  end
end
