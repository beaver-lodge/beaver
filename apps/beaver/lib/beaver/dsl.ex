defmodule Beaver.DSL do
  def transform_ssa(block) do
    Macro.prewalk(block, fn
      # one SSA
      {:"::", _,
       [
         {:=, line2,
          [
            var,
            {mf, line3, args_ast}
          ]},
         sigil_t
       ]} ->
        arguments_ast =
          case args_ast do
            [arg_list] when is_list(arg_list) ->
              quote do
                unquote(arg_list) ++ [result_types: unquote(sigil_t)]
              end

            _ ->
              quote do
                [unquote_splicing(args_ast), result_types: unquote(sigil_t)]
              end
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
         sigil_t
       ]} ->
        quote do
          {unquote(var), unquote(sigil_t)}
        end

      #  expression with no binding
      {:"::", _,
       [
         {mf, line2, args_ast},
         sigil_t = {:sigil_t, _, _}
       ]}
      when is_list(args_ast) ->
        case args_ast do
          [args, [do: ast_block]] when is_list(args) ->
            {mf, line2,
             quote do
               [unquote(args) ++ [result_types: unquote(sigil_t)], [do: unquote(ast_block)]]
             end}

          [args] when is_list(args) ->
            {mf, line2,
             [
               quote do
                 unquote(args) ++ [result_types: unquote(sigil_t)]
               end
             ]}

          all_args when is_list(all_args) and length(all_args) >= 2 ->
            {others, _} = Enum.split(all_args, -1)
            last = List.last(all_args)

            # if last arg is a keyword, join them all
            if is_list(last) do
              {mf, line2,
               [
                 quote do
                   [
                     unquote_splicing(others),
                     unquote_splicing(last),
                     result_types: unquote(sigil_t)
                   ]
                 end
               ]}
            else
              {mf, line2,
               [
                 quote do
                   [unquote_splicing(args_ast), result_types: unquote(sigil_t)]
                 end
               ]}
            end

          _ ->
            {mf, line2,
             [
               quote do
                 [unquote_splicing(args_ast), result_types: unquote(sigil_t)]
               end
             ]}
        end

      other ->
        other
    end)
  end
end
