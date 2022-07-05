defmodule Beaver.MLIR do
  defmacro block(call, do: block) do
    {
      _block_args,
      _block_opts,
      args_type_ast,
      args_var_ast,
      locations_var_ast,
      block_arg_var_ast
    } = Beaver.MLIR.DSL.Block.transform_call(call)

    {block_id, _} = Macro.decompose_call(call)
    if not is_atom(block_id), do: raise("block name must be an atom")

    block_ast =
      quote do
        unquote_splicing(args_type_ast)
        block_arg_types = [unquote_splicing(args_var_ast)]
        block_arg_locs = [unquote_splicing(locations_var_ast)]

        block = Beaver.MLIR.Block.create(block_arg_types, block_arg_locs)

        # can't put code here inside a function like Region.under, because we need to support uses across blocks
        previous_block = Beaver.MLIR.Managed.Block.get()

        Beaver.MLIR.Managed.Block.set(block)

        if region = Beaver.MLIR.Managed.Region.get() do
          # insert the block to region
          Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
          # put the block to managed terminator => block id (name in decomposed block call)
          Beaver.MLIR.Managed.Terminator.put_block(unquote(block_id), block)
        else
          raise "no managed region found to append block"
        end

        unquote_splicing(block_arg_var_ast)
        block_ret = unquote(block)
        Beaver.MLIR.Managed.Block.set(previous_block)

        block_ret
      end

    block_ast
  end

  # TODO: check sigil_t is from MLIR
  defmacro region(do: block) do
    quote do
      region = Beaver.MLIR.CAPI.mlirRegionCreate()

      Beaver.MLIR.Region.under(region, fn ->
        unquote(block)
      end)

      [region]
    end
  end
end
