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
       ]}
      when is_list(args_ast) ->
        arguments_ast =
          quote do
            [unquote_splicing(args_ast), result_types: unquote(sigil_t)]
          end

        {mf, line2, [arguments_ast]}

      other ->
        other
    end)
  end
end
