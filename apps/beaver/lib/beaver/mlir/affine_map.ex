defmodule Beaver.MLIR.AffineMap do
  alias Beaver.MLIR

  def create(dim_cnt, symbol_cnt, exprs, opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)

    exprs =
      exprs
      |> Enum.map(fn
        const when is_integer(const) -> MLIR.CAPI.mlirAffineConstantExprGet(ctx, const)
        expr -> expr
      end)

    expr_array = MLIR.CAPI.MlirAffineExpr.array(exprs, mut: true)
    MLIR.CAPI.mlirAffineMapGet(ctx, dim_cnt, symbol_cnt, length(exprs), expr_array)
  end

  def dim(index, opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    MLIR.CAPI.mlirAffineDimExprGet(ctx, index)
  end

  def symbol(index, opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    MLIR.CAPI.mlirAffineSymbolExprGet(ctx, index)
  end
end
