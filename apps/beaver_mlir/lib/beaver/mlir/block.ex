defmodule Beaver.MLIR.Block do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI.IR

  # TODO: remote ctx in these funcs

  # TODO: use the struct to replace the Exotic.Value here in pattern after Exotic gets updated with Protocol support
  def do_add_arg!(block, _ctx, {t = %Exotic.Value{}, loc}) do
    IR.mlirBlockAddArgument(block, t, loc)
  end

  def do_add_arg!(block, ctx, {t, loc}) do
    t = IR.mlirTypeParseGet(ctx, IR.string_ref(t))
    IR.mlirBlockAddArgument(block, t, loc)
  end

  def do_add_arg!(block, ctx, t) do
    loc = IR.mlirLocationUnknownGet(ctx)
    t = IR.mlirTypeParseGet(ctx, IR.string_ref(t))
    IR.mlirBlockAddArgument(block, t, loc)
  end

  def add_arg!(block, ctx, args) do
    for arg <- args do
      do_add_arg!(block, ctx, arg)
    end
  end

  def get_arg!(block, index) do
    IR.mlirBlockGetArgument(block, index) |> Exotic.Value.transmit()
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
  run function f and append ops created to block. This function only works when there is no uses across blocks. For instance, a ModuleOp has only one region and one block.
  """
  def under(block, f) when is_function(f, 0) do
    previous_block = Beaver.MLIR.Managed.Block.get()
    Beaver.MLIR.Managed.Block.set(block)
    last_op = f.()
    Beaver.MLIR.Managed.Block.set(previous_block)
    last_op
  end
end
