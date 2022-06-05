defmodule Beaver.MLIR.Dialect.Builtin do
  defmacro module(call, do: block) do
    quote do
    end
  end

  defmacro module(do: block) do
    # TODO: check sigil_t is from MLIR
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
      location = Beaver.MLIR.Managed.Location.get()
      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)
      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)

      __module__insert_point__ = fn op ->
        Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(module_body_block, op)
      end

      Beaver.MLIR.Managed.InsertionPoint.push(__module__insert_point__)
      unquote(new_block_ast)
      Beaver.MLIR.Managed.InsertionPoint.pop()

      Beaver.MLIR.Managed.Block.clear_ids()

      if not Beaver.MLIR.Managed.InsertionPoint.empty?(),
        do: raise("insertion point should be cleared")

      module
    end
  end
end
