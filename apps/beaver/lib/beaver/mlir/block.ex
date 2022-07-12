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

  def create(args, locs) when is_list(args) and is_list(locs) do
    # TODO: improve this
    if length(args) != length(locs) do
      raise "Different length of block args and types. Make sure the block/1 macro in call within mlir/1 macro"
    end

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

  def is_null(block = %MLIR.CAPI.MlirBlock{}) do
    block
    |> Exotic.Value.fetch(MLIR.CAPI.MlirBlock, :ptr)
    |> Exotic.Value.extract() == 0
  end

  defp clone_op(from: concrete, to: values) when is_list(values) do
    values
  end

  defp clone_op(from: from, to: new_op) do
    # %concrete{} = Beaver.concrete(from)

    # MLIR.CAPI.mlirOperationEqual(from, new_op)
    # |> Exotic.Value.extract()
    # |> IO.inspect(label: "mlirOperationEqual #{concrete}")

    from
  end

  @doc """
  Clone a block with ops might be rewritten with new ops or values.
  """
  @spec clone(MLIR.CAPI.MlirBlock.t(), [MLIR.CAPI.MlirOperation.t() | [MLIR.CAPI.MlirValue.t()]]) ::
          MLIR.CAPI.MlirBlock.t()
  def clone(%MLIR.CAPI.MlirBlock{} = block, ops) do
    use Beaver

    Enum.zip(Beaver.Walker.operations(block), ops)
    |> Enum.map(fn {op, new_op} ->
      clone_op(from: op, to: new_op)
    end)

    block
  end
end
