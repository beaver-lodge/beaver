defmodule Beaver.MLIR.Block do
  alias Beaver.MLIR
  require Beaver.MLIR.CAPI
  # TODO: remote ctx in these funcs

  def do_add_arg!(block, _ctx, {t = %Beaver.MLIR.CAPI.MlirType{}, loc}) do
    MLIR.CAPI.mlirBlockAddArgument(block, t, loc)
  end

  def do_add_arg!(block, ctx, {t, loc}) do
    t = MLIR.CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(t))
    MLIR.CAPI.mlirBlockAddArgument(block, t, loc)
  end

  def do_add_arg!(block, ctx, t) do
    loc = MLIR.CAPI.mlirLocationUnknownGet(ctx)
    t = MLIR.CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(t))
    MLIR.CAPI.mlirBlockAddArgument(block, t, loc)
  end

  def add_arg!(block, ctx, args) do
    for arg <- args do
      do_add_arg!(block, ctx, arg)
    end
  end

  def get_arg!(block, index) when not is_nil(block) do
    MLIR.CAPI.mlirBlockGetArgument(block, index)
  end

  def create(arg_loc_pairs) when is_list(arg_loc_pairs) do
    {args, locs} =
      Enum.reduce(arg_loc_pairs, {[], []}, fn {arg, loc}, {args, locs} ->
        {args ++ [arg], locs ++ [loc]}
      end)

    create(args, locs)
  end

  def create(args, locs) when is_list(args) and is_list(locs) do
    # TODO: improve this
    if length(args) != length(locs) do
      raise "Different length of block args and types. Make sure the block/1 macro in call within mlir/1 macro"
    end

    len = length(args)
    args = args |> MLIR.CAPI.array(MLIR.CAPI.MlirType)
    locs = locs |> MLIR.CAPI.array(MLIR.CAPI.MlirLocation)

    MLIR.CAPI.mlirBlockCreate(
      len,
      args,
      locs
    )
  end
end
