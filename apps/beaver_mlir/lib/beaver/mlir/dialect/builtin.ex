defmodule Beaver.MLIR.Dialect.Builtin do
  defmacro module(call, do: block) do
    IO.inspect(call)
    IO.inspect(block)

    quote do
    end
  end

  defmacro module(do: block) do
    block |> IO.inspect()

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
          return_type_ast =
            quote do
              [return_type: unquote(sigil_t)]
            end

          # TODO, check if last arg is keyword
          {:=, line2,
           [
             var,
             {mf, line3, args_ast ++ [return_type_ast]}
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
          return_type_ast =
            quote do
              [return_type: unquote(sigil_t)]
            end

          {mf, line2, args_ast ++ [return_type_ast]}

        other ->
          other
      end)

    new_block_ast |> Macro.to_string() |> IO.puts()

    new_block_ast
  end
end
