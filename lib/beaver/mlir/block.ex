defmodule Beaver.MLIR.Block do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  require Beaver.MLIR.CAPI

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  defp do_add_args!(block, ctx, {t, loc}) when is_function(t, 1) or is_function(loc, 1) do
    MLIR.CAPI.mlirBlockAddArgument(
      block,
      t |> Beaver.Deferred.create(ctx),
      loc |> Beaver.Deferred.create(ctx)
    )
  end

  defp do_add_args!(block, ctx, {t = %Beaver.MLIR.Type{}, loc}) do
    loc = loc |> Beaver.Deferred.create(ctx)
    MLIR.CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, {t = {:parametric, _, _, _f}, loc}) do
    t = Beaver.Deferred.create(t, ctx)
    MLIR.CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, {t, loc}) do
    t = MLIR.CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(t))
    MLIR.CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, t) do
    loc = MLIR.CAPI.mlirLocationUnknownGet(ctx)
    do_add_args!(block, ctx, {t, loc})
  end

  def add_args!(block, args, opts \\ []) when is_list(args) do
    ctx =
      opts[:ctx] ||
        Enum.find_value(args, fn
          t = %Beaver.MLIR.Type{} -> MLIR.CAPI.mlirTypeGetContext(t)
          _ -> nil
        end)

    unless ctx do
      raise "requires a MLIR Context to add args, or types already being created so the the context could be extracted from them"
    end

    for arg <- args do
      do_add_args!(block, ctx, arg)
    end
  end

  def get_arg!(block, index) when not is_nil(block) do
    MLIR.CAPI.mlirBlockGetArgument(block, index)
  end

  def create(arg_loc_pairs \\ []) when is_list(arg_loc_pairs) do
    {args, locs} =
      Enum.reduce(arg_loc_pairs, {[], []}, fn {arg, loc}, {args, locs} ->
        {args ++ [arg], locs ++ [loc]}
      end)

    create(args, locs)
  end

  def create(args, locs) when is_list(args) and is_list(locs) do
    if length(args) != length(locs) do
      raise "Different length of block args and types. Make sure the block/1 macro in call within mlir/1 macro"
    end

    len = length(args)
    args = args |> Beaver.Native.array(MLIR.Type)
    locs = locs |> Beaver.Native.array(MLIR.Location)

    MLIR.CAPI.mlirBlockCreate(
      len,
      args,
      locs
    )
  end
end
