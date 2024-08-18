defmodule Beaver.MLIR.Pattern do
  alias Beaver.MLIR

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
    pattern_module = MLIR.Module.create(ctx, "")

    for p <- patterns do
      p = p.(pattern_module)

      if opts[:debug] do
        p |> MLIR.dump!()
      end
    end

    pattern_set =
      MLIR.CAPI.mlirPDLPatternModuleFromModule(pattern_module)
      |> MLIR.CAPI.mlirRewritePatternSetFromPDLPatternModule()
      |> MLIR.CAPI.mlirFreezeRewritePattern()

    MLIR.PatternSet.apply_(op, pattern_set)
  end
end
