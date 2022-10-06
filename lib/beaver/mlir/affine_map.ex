defmodule Beaver.MLIR.AffineMap do
  alias Beaver.MLIR

  def create(dim_cnt, symbol_cnt, exprs, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        exprs =
          exprs
          |> Enum.map(fn
            const when is_integer(const) -> MLIR.CAPI.mlirAffineConstantExprGet(ctx, const)
            f when is_function(f, 1) -> f.(ctx)
            expr -> expr
          end)

        expr_array = MLIR.CAPI.MlirAffineExpr.array(exprs, mut: true)
        MLIR.CAPI.mlirAffineMapGet(ctx, dim_cnt, symbol_cnt, length(exprs), expr_array)
      end
    )
  end

  def dim(index, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &MLIR.CAPI.mlirAffineDimExprGet(&1, index)
    )
  end

  def symbol(index, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &MLIR.CAPI.mlirAffineSymbolExprGet(&1, index)
    )
  end
end