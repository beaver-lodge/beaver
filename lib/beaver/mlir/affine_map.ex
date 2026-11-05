alias Beaver.MLIR.CAPI
alias Beaver.Deferred

defmodule Beaver.MLIR.AffineMap do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Beaver.Native

  def create(dim_cnt, symbol_cnt, exprs, opts \\ []) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        exprs =
          exprs
          |> Enum.map(fn
            const when is_integer(const) -> CAPI.mlirAffineConstantExprGet(ctx, const)
            %Deferred{} = deferred -> Deferred.resolve(deferred, ctx)
            expr -> expr
          end)

        expr_array = Beaver.Native.array(exprs, MLIR.AffineExpr, mut: true)
        CAPI.mlirAffineMapGet(ctx, dim_cnt, symbol_cnt, length(exprs), expr_array)
      end
    )
  end

  def dim(index, opts \\ []), do: Deferred.from_opts(opts, &CAPI.mlirAffineDimExprGet(&1, index))

  def symbol(index, opts \\ []),
    do: Deferred.from_opts(opts, &CAPI.mlirAffineSymbolExprGet(&1, index))
end
