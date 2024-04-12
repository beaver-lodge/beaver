defmodule Beaver.MIF.Intrinsic do
  defmacro __using__(_opts) do
    quote do
      use Beaver
      import Beaver.MIF.Intrinsic
    end
  end

  defmacro defi(call, do_block) do
    {call, opts} =
      Macro.prewalk(call, nil, fn
        ast = {:when, _, _}, acc ->
          {ast, acc}

        ast, nil ->
          {name, args} = Macro.decompose_call(ast)
          opts = List.last(args)
          args = Enum.take(args, length(args) - 1)

          {quote(do: handle_intrinsic(unquote(name), [unquote_splicing(args)], unquote(opts))),
           opts}

        ast, acc ->
          {ast, acc}
      end)

    quote do
      def unquote(call) do
        mlir(
          [ctx: unquote(opts)[:ctx], block: unquote(opts)[:block]],
          unquote(do_block)
        )
      end
    end
  end
end
