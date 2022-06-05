defmodule Beaver.MLIR do
  defmacro block(call, do: block) do
    {
      block_args,
      block_opts,
      args_type_ast,
      args_var_ast,
      locations_var_ast,
      block_arg_var_ast
    } = Beaver.MLIR.DSL.Block.transform_call(call)

    {block_id, _} = Macro.decompose_call(call)
    if not is_atom(block_id), do: raise("block name must be an atom")

    block_arg_var_ast |> Macro.to_string()

    block_ast =
      quote do
        unquote_splicing(args_type_ast)
        block_arg_types = [unquote_splicing(args_var_ast)]
        block_arg_locs = [unquote_splicing(locations_var_ast)]

        block = Beaver.MLIR.Block.create(block_arg_types, block_arg_locs)
        Beaver.MLIR.Managed.Block.push(unquote(block_id), block)

        unquote_splicing(block_arg_var_ast)

        Beaver.MLIR.Managed.InsertionPoint.push(fn op ->
          Beaver.MLIR.CAPI.mlirBlockInsertOwnedOperation(block, 0, op)
        end)

        unquote(block)

        Beaver.MLIR.Managed.InsertionPoint.pop()

        Beaver.MLIR.Managed.Region.get()
        |> Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(Beaver.MLIR.Managed.Block.pop())
      end

    block_ast
  end

  # TODO: check sigil_t is from MLIR
  defmacro region(do: block) do
    new_block_ast =
      Macro.prewalk(block, fn
        # one SSA
        {:"::", _,
         [
           {:=, line2,
            [
              var,
              {mf, line3, args_ast}
            ]},
           sigil_t = {:sigil_t, _, _}
         ]} ->
          arguments_ast =
            quote do
              [unquote_splicing(args_ast), result_types: unquote(sigil_t)]
            end

          # TODO, check if last arg is keyword
          {:=, line2,
           [
             var,
             {mf, line3, [arguments_ast]}
           ]}

        #  block arg
        {:"::", _,
         [
           var = {_var_name, _, nil},
           sigil_t = {:sigil_t, _, _}
         ]} ->
          quote do
            {unquote(var), unquote(sigil_t)}
          end

        #  expression with no binding
        {:"::", _,
         [
           {mf, line2, args_ast},
           sigil_t = {:sigil_t, _, _}
         ]} ->
          arguments_ast =
            quote do
              [unquote_splicing(args_ast), result_types: unquote(sigil_t)]
            end

          {mf, line2, [arguments_ast]}

        other ->
          other
      end)

    new_block_ast |> Macro.to_string() |> IO.puts()

    quote do
      func_region = Beaver.MLIR.CAPI.mlirRegionCreate()
      Beaver.MLIR.Managed.Region.set(func_region)
      unquote(new_block_ast)
      Beaver.MLIR.Managed.Terminator.resolve()
    end
  end
end
