defmodule Beaver.DSL do
  @moduledoc false
  def transform_ssa(block) do
    Macro.prewalk(block, fn
      #  block arg
      {:"::", _,
       [
         var = {_var_name, _, nil},
         sigil_t
       ]} ->
        quote do
          {unquote(var), unquote(sigil_t)}
        end

      other ->
        other
    end)
  end
end
