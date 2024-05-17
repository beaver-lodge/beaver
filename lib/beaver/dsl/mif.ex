defmodule Beaver.MIF do
  @doc """
  A MIF is NIF generated by LLVM/MLIR
  """
  require Beaver.Env
  alias Beaver.MLIR.{Type, Attribute}
  alias Beaver.MLIR.Dialect.{Arith, LLVM, Func, CF}
  require Func
  use Beaver

  defmacro __using__(_opts) do
    quote do
      import Beaver.MIF
      use Beaver
      require Beaver.MLIR.Dialect.Func
      alias Beaver.MLIR.Dialect.{Func, Arith, LLVM, CF}
      alias MLIR.{Type, Attribute}
      import Type

      @before_compile Beaver.MIF
      Module.register_attribute(__MODULE__, :defm, accumulate: true)
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      @ir @defm |> Enum.reverse() |> Beaver.MIF.compile_definitions()
      def __ir__ do
        @ir
      end
    end
  end

  def compile_definitions(definitions) do
    import Beaver.MLIR.Transforms
    ctx = Beaver.MLIR.Context.create()
    available_ops = MapSet.new(MLIR.Dialect.Registry.ops(:all, ctx: ctx))

    m =
      mlir ctx: ctx do
        module do
          mlir = %{
            ctx: ctx,
            blk: Beaver.Env.block(),
            available_ops: available_ops,
            vars: Map.new(),
            region: nil
          }

          for {env, d} <- definitions do
            {call, ret_types, body} = d

            ast =
              quote do
                def(unquote(call) :: unquote(ret_types), unquote(body))
              end

            # |> tap(fn ast -> ast |> Macro.to_string() |> IO.puts() end)

            Beaver.MIF.Expander.expand_with_mlir(
              ast,
              mlir,
              env
            )
          end
        end
      end
      |> MLIR.Pass.Composer.nested(
        "func.func",
        Beaver.MIF.Pass.CreateAbsentFunc
      )
      |> canonicalize
      |> MLIR.Pass.Composer.run!(print: System.get_env("DEFM_PRINT_IR") == "1")
      |> MLIR.to_string(bytecode: true)

    MLIR.Context.destroy(ctx)
    m
  end

  def mangling(mod, func) do
    Module.concat(mod, func)
  end

  defmacro op(_), do: :implemented_in_expander
  defmacro call({:"::", _, [_call, _types]}), do: :implemented_in_expander
  defmacro call(_mod, {:"::", _, [_call, _types]}), do: :implemented_in_expander

  defmacro for_loop(_expr, do: _body), do: :implemented_in_expander

  defmacro while_loop(_expr, do: _body), do: :implemented_in_expander

  defmacro cond_br(_condition, _clauses), do: :implemented_in_expander

  defmacro struct_if(_condition, _clauses), do: :implemented_in_expander

  defmacro value(_expr), do: :implemented_in_expander

  def decompose_call_and_returns(call) do
    case call do
      {:"::", _, [call, ret_type]} -> {call, [ret_type]}
      call -> {call, []}
    end
  end

  def normalize_call(call) do
    {name, args} = Macro.decompose_call(call)

    args =
      for i <- Enum.with_index(args) do
        case i do
          # env
          {a = {:env, _, nil}, 0} ->
            quote do
              unquote(a) :: Beaver.MIF.Env.t()
            end

          # term
          {a = {name, _, context}, version}
          when is_atom(name) and is_atom(context) and is_integer(version) ->
            quote do
              unquote(a) :: Beaver.MIF.Term.t()
            end

          # typed
          {at = {:"::", _, [_a, _t]}, _} ->
            at
        end
      end

    quote do
      unquote(name)(unquote_splicing(args))
    end
  end

  defmacro defm(call, body \\ []) do
    {call, ret_types} = decompose_call_and_returns(call)

    call = normalize_call(call)
    {name, args} = Macro.decompose_call(call)
    env = __CALLER__
    [_enif_env | invoke_args] = args

    invoke_args =
      for {:"::", _, [a, _t]} <- invoke_args do
        a
      end

    quote do
      @defm unquote(Macro.escape({env, {call, ret_types, body}}))
      def unquote(name)(unquote_splicing(invoke_args)) do
        f = &Beaver.MIF.JIT.invoke(&1, {unquote(env.module), unquote(name), unquote(invoke_args)})

        if jit = Beaver.MIF.JIT.get(__MODULE__) do
          f.(jit)
        else
          f
        end
      end
    end
  end
end
