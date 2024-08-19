defmodule Beaver.MLIR.Pattern do
  alias Beaver.MLIR
  import MLIR.CAPI

  @apply_default_opts [debug: false]
  @doc """
  Apply patterns on a container (region, operation, module).
  It returns the container if it succeeds otherwise it raises.
  """
  def apply!(op, patterns, opts \\ @apply_default_opts) do
    case apply_(op, patterns, opts) do
      {:ok, module} ->
        module

      _ ->
        raise "failed to apply pattern"
    end
  end

  @doc """
  Apply patterns on a container (region, operation, module).
  It is named `apply_` with a underscore to avoid name collision with `Kernel.apply/2`
  """
  def apply_(op, patterns, opts \\ @apply_default_opts) when is_list(patterns) do
    if MLIR.is_null(op), do: raise("op is null")
    ctx = MLIR.Operation.from_module(op) |> mlirOperationGetContext()
    pattern_module = MLIR.Location.from_env(__ENV__, ctx: ctx) |> MLIR.Module.empty()

    for p <- patterns do
      p = p.(pattern_module)

      if opts[:debug] do
        p |> MLIR.dump!()
      end
    end

    MLIR.Operation.verify!(pattern_module)
    MLIR.Operation.verify!(op)
    pdl_pat_mod = mlirPDLPatternModuleFromModule(pattern_module)

    frozen_pat_set =
      pdl_pat_mod |> mlirRewritePatternSetFromPDLPatternModule() |> mlirFreezeRewritePattern()

    result = beaverApplyPatternsAndFoldGreedily(op, frozen_pat_set)
    mlirPDLPatternModuleDestroy(pdl_pat_mod)
    mlirFrozenRewritePatternSetDestroy(frozen_pat_set)
    MLIR.Module.destroy(pattern_module)

    if MLIR.LogicalResult.success?(result) do
      {:ok, op}
    else
      {:error, "failed to apply pattern set"}
    end
  end
end
