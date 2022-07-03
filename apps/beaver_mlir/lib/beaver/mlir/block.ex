defmodule Beaver.MLIR.Block do
  alias Beaver.MLIR

  # TODO: remote ctx in these funcs

  # TODO: use the struct to replace the Exotic.Value here in pattern after Exotic gets updated with Protocol support
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
    MLIR.CAPI.mlirBlockGetArgument(block, index) |> Exotic.Value.transmit()
  end

  def create(arg_loc_pairs) when is_list(arg_loc_pairs) do
    {args, locs} =
      Enum.reduce(arg_loc_pairs, {[], []}, fn {arg, loc}, {args, locs} ->
        {args ++ [arg], locs ++ [loc]}
      end)

    create(args, locs)
  end

  def create(args, locs) when length(args) == length(locs) do
    len = length(args)
    args = args |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr()
    locs = locs |> Exotic.Value.Array.get() |> Exotic.Value.get_ptr()

    MLIR.CAPI.mlirBlockCreate(
      len,
      args,
      locs
    )
  end

  @doc """
  run function f and append ops created to block. This function only works when there is no uses across blocks. For instance, a ModuleOp has only one region of one block.
  """
  def under(block, f) when is_function(f, 0) do
    previous_block = Beaver.MLIR.Managed.Block.get()
    Beaver.MLIR.Managed.Block.set(block)
    last_op = f.()
    Beaver.MLIR.Managed.Block.set(previous_block)
    last_op
  end
end
