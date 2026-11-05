alias Beaver.MLIR.{CAPI, Location, Type}
alias Beaver.{Deferred, Native}

defmodule Beaver.MLIR.Block do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native

  defp do_add_args!(block, ctx, {t, loc}) when is_struct(t, Deferred) or is_struct(loc, Deferred),
    do: CAPI.mlirBlockAddArgument(block, t |> Deferred.resolve(ctx), loc |> Deferred.resolve(ctx))

  defp do_add_args!(block, ctx, {t = %Type{}, loc}) do
    t = t |> Deferred.resolve(ctx)
    loc = loc |> Deferred.resolve(ctx)
    CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, {t = {:parametric, _, _, _f}, loc}) do
    t = Deferred.resolve(t, ctx)
    loc = Deferred.resolve(loc, ctx)
    CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, {t, loc}) do
    t = CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(t))
    loc = Deferred.resolve(loc, ctx)
    CAPI.mlirBlockAddArgument(block, t, loc)
  end

  defp do_add_args!(block, ctx, t) do
    loc = CAPI.mlirLocationUnknownGet(ctx)
    do_add_args!(block, ctx, {t, loc})
  end

  @type arg_type :: Type.t() | String.t()
  @type arg :: {arg_type(), Location.t()} | arg_type()
  @spec add_args!(__MODULE__.t(), list(arg), Keyword.t()) :: any()
  @doc """
  add arguments to a block
  """
  def add_args!(block, args, opts \\ []) when is_list(args) do
    ctx =
      Deferred.context(opts) ||
        Enum.find_value(args, fn
          t = %Type{} -> MLIR.context(t)
          {t = %Type{}, _} -> MLIR.context(t)
          {_, l = %Location{}} -> MLIR.context(l)
          _ -> nil
        end)

    if !ctx, do: raise("Requires MLIR Context to add arguments. Otherwise, use types or locations already created.")

    for arg <- args do
      do_add_args!(block, ctx, arg)
    end
  end

  def get_arg!(%__MODULE__{} = block, index), do: CAPI.mlirBlockGetArgument(block, index)

  def create(arg_loc_pairs \\ []) when is_list(arg_loc_pairs) do
    {args, locs} =
      Enum.reduce(arg_loc_pairs, {[], []}, fn {arg, loc}, {args, locs} -> {args ++ [arg], locs ++ [loc]} end)

    create(args, locs)
  end

  def create(args, locs) when is_list(args) and is_list(locs) do
    if length(args) != length(locs),
      do: raise("Different length of block args and types. Make sure the block/1 macro in call within mlir/1 macro")

    len = length(args)
    args = args |> Native.array(Type)
    locs = locs |> Native.array(Location)

    CAPI.mlirBlockCreate(len, args, locs)
  end

  defdelegate destroy(blk), to: CAPI, as: :mlirBlockDestroy
  defdelegate append(blk, op), to: CAPI, as: :mlirBlockAppendOwnedOperation
end
