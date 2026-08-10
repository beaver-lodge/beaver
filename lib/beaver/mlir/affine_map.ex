defmodule Beaver.MLIR.AffineMap do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  def create(dim_cnt, symbol_cnt, exprs, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        exprs =
          exprs
          |> Enum.map(fn
            const when is_integer(const) -> MLIR.CAPI.mlirAffineConstantExprGet(ctx, const)
            %Beaver.Deferred{} = deferred -> Beaver.Deferred.resolve(deferred, ctx)
            expr -> expr
          end)

        expr_array = Beaver.Native.array(exprs, MLIR.AffineExpr, mut: true)
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
