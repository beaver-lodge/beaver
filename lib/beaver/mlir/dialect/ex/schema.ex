defmodule Beaver.MLIR.Dialect.Ex.Schema do
  @moduledoc false
  @pure ~w(lit add sub mul cmp box to_word unbox)a

  def pure_effects(_op), do: :pure
  def speculatable(_op), do: :speculatable

  defmacro defop(call, opts \\ []) do
    # Fragment variables name schema operands and constraints in the facade.
    {call, opts} =
      Macro.prewalk({call, opts}, fn
        {name, meta, context} when is_atom(name) and is_atom(context) -> {name, meta, nil}
        ast -> ast
      end)

    {name, _} = Macro.decompose_call(call)

    opts =
      if name in @pure do
        Keyword.put(opts, :interfaces,
          memory_effects: quote(do: &unquote(__MODULE__).pure_effects/1),
          conditionally_speculatable: quote(do: &unquote(__MODULE__).speculatable/1)
        )
      else
        opts
      end

    quote do
      @ex_operations {unquote("ex.#{name}"), @ex_layer}
      Beaver.Slang.defop(unquote(call), unquote(opts))
    end
  end
end
