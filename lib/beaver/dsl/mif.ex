defmodule Beaver.MIF do
  @doc """
  A MIF is NIF generated by LLVM/MLIR
  """
  require Beaver.Env
  use Beaver

  defmacro __using__(_opts) do
    quote do
      import Beaver.MIF
      @before_compile Beaver.MIF
      Module.register_attribute(__MODULE__, :defm, accumulate: true)
    end
  end

  defmacro __before_compile__(env) do
    quote do
      {ir_str, _} = @defm |> Beaver.MIF.compile_definitions(__ENV__) |> Code.eval_quoted()
      @ir ir_str
      def __ir__ do
        @ir
      end
    end
  end

  defp transform_type_operator({:"::", _, [var, t]}) do
    {var, t}
  end

  def compile_definitions(definitions, env) do
    functions =
      for {call, expr} <- definitions do
        {name, args} = Macro.decompose_call(call)
        expr = Macro.postwalk(expr, &Macro.expand(&1, env)) |> dbg

        quote do
          Beaver.MLIR.Dialect.Func.func unquote(name)(
                                          function_type: Type.function([Type.i32()], [Type.i32()])
                                        ) do
            # unquote(compile_args(call))
            # unquote(expr)
            region do
              block _() do
                Beaver.MLIR.Dialect.Func.return() >>> []
              end
            end
          end
        end
      end

    quote do
      ctx = Beaver.MLIR.Context.create()

      m =
        mlir ctx: ctx do
          module do
            require Beaver.MLIR.Dialect.Func
            alias MLIR.Type
            (unquote_splicing(functions))
          end
        end
        |> MLIR.to_string()

      MLIR.Context.destroy(ctx)
      m
    end
    |> tap(&IO.puts(Macro.to_string(&1)))
  end

  defmacro op(
             {:"::", _,
              [
                call,
                types
              ]},
             block \\ []
           ) do
    "??"
  end

  defmacro defm(call, expr \\ []) do
    {name, args} = Macro.decompose_call(call)
    args = for {:"::", _, [arg, _type]} <- args, do: arg

    quote do
      @defm unquote(Macro.escape({call, expr}))
      def unquote(name)(unquote_splicing(args)) do
        arguments = [unquote_splicing(args)] |> Enum.map(&Beaver.Native.I64.make/1)
        __ir__()
      end
    end
  end
end
