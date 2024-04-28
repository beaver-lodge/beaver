defmodule Beaver.MIF do
  @doc """
  A MIF is NIF generated by LLVM/MLIR
  """
  require Beaver.Env
  alias Beaver.MLIR.{Type, Attribute}
  alias Beaver.MLIR.Dialect.{Arith, LLVM, Func, CF}
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

  defp inject_mlir_opts({:^, _, [{_, _, _} = var]}) do
    quote do
      CF.br({unquote(var), []}) >>> []
    end
  end

  @intrinsics Beaver.MIF.Prelude.intrinsics()

  defp inject_mlir_opts({f, _, args}) when f in @intrinsics do
    quote do
      Beaver.MIF.Prelude.handle_intrinsic(unquote(f), [unquote_splicing(args)],
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block()
      )
    end
  end

  defp inject_mlir_opts(ast) do
    with {{:__aliases__, _, _} = m, f, args} <- Macro.decompose_call(ast) do
      quote do
        if function_exported?(unquote(m), :handle_intrinsic, 3) or
             macro_exported?(unquote(m), :handle_intrinsic, 3) do
          unquote(m).handle_intrinsic(unquote(f), [unquote_splicing(args)],
            ctx: Beaver.Env.context(),
            block: Beaver.Env.block()
          )
        else
          unquote(ast)
        end
      end
    else
      :error ->
        ast

      _ ->
        ast
    end
  end

  defp definition_to_func(env, {call, ret_types, body}) do
    {name, args} = Macro.decompose_call(call)
    name = mangling(env.module, name)

    body =
      body[:do]
      |> Macro.postwalk(&inject_mlir_opts(&1))
      |> List.wrap()

    {args, arg_types} =
      for {:"::", _, [a, t]} <- args do
        {a, t}
      end
      |> Enum.unzip()

    arg_types = Enum.map(arg_types, &inject_mlir_opts/1)
    ret_types = Enum.map(ret_types, &inject_mlir_opts/1)

    quote do
      mlir ctx: var!(ctx), block: var!(block) do
        num_of_args = unquote(length(args))

        ret_types =
          unquote(ret_types) |> Enum.map(&Beaver.Deferred.create(&1, Beaver.Env.context()))

        arg_types =
          unquote(arg_types) |> Enum.map(&Beaver.Deferred.create(&1, Beaver.Env.context()))

        Beaver.MLIR.Dialect.Func.func unquote(name)(
                                        function_type: Type.function(arg_types, ret_types)
                                      ) do
          region do
            block _entry() do
              MLIR.Block.add_args!(Beaver.Env.block(), arg_types, ctx: Beaver.Env.context())

              [unquote_splicing(args)] =
                Range.new(0, num_of_args - 1)
                |> Enum.map(&MLIR.Block.get_arg!(Beaver.Env.block(), &1))

              unquote(body)
            end
          end
        end
      end
    end
  end

  def compile_definitions(definitions) do
    import Beaver.MLIR.Transforms
    ctx = Beaver.MLIR.Context.create()

    m =
      mlir ctx: ctx do
        module do
          for {env, d} <- definitions do
            f = definition_to_func(env, d)
            binding = [ctx: Beaver.Env.context(), block: Beaver.Env.block()]
            Code.eval_quoted(f, binding, env)
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

  defmacro op({:"::", _, [call, types]}, _ \\ []) do
    {{:., _, [{dialect, _meta, nil}, op]}, _, args} = call

    quote do
      %Beaver.SSA{
        op: unquote("#{dialect}.#{op}"),
        arguments: List.flatten(unquote(args)),
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block(),
        loc: Beaver.MLIR.Location.from_env(unquote(Macro.escape(__CALLER__)))
      }
      |> Beaver.SSA.put_results([unquote_splicing(List.wrap(types))])
      |> MLIR.Operation.create()
    end
  end

  def mangling(mod, func) do
    Module.concat(mod, func)
  end

  defmacro call({:"::", _, [call, types]}) do
    quote do
      call(__MODULE__, unquote(call) :: unquote(types))
    end
  end

  defmacro call(mod, {:"::", _, [call, types]}) do
    {name, args} = Macro.decompose_call(call)

    quote do
      name = Beaver.MIF.mangling(unquote(mod), unquote(name))

      %Beaver.SSA{
        op: unquote("func.call"),
        arguments: [unquote_splicing(args), callee: Attribute.flat_symbol_ref("#{name}")],
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block(),
        loc: Beaver.MLIR.Location.from_env(unquote(Macro.escape(__CALLER__)))
      }
      |> Beaver.SSA.put_results([unquote_splicing(List.wrap(types))])
      |> MLIR.Operation.create()
    end
  end

  defmacro for_loop(expr, do: body) do
    {:<-, _, [{element, index}, {:{}, _, [t, ptr, len]}]} = expr

    quote do
      mlir do
        alias Beaver.MLIR.Dialect.{Index, SCF, LLVM}
        zero = Index.constant(value: Attribute.index(0)) >>> Type.index()
        lower_bound = zero
        upper_bound = Index.casts(unquote(len)) >>> Type.index()
        step = Index.constant(value: Attribute.index(1)) >>> Type.index()

        SCF.for [lower_bound, upper_bound, step] do
          region do
            block _body(unquote(index) >>> Type.index()) do
              index_casted = Index.casts(unquote(index)) >>> Type.i64()

              element_ptr =
                LLVM.getelementptr(unquote(ptr), index_casted,
                  elem_type: unquote(t),
                  rawConstantIndices: ~a{array<i32: -2147483648>}
                ) >>> ~t{!llvm.ptr}

              var!(unquote(element)) = LLVM.load(element_ptr) >>> unquote(t)
              unquote(body)
              Beaver.MLIR.Dialect.SCF.yield() >>> []
            end
          end
        end >>> []
      end
    end
  end

  defmacro while_loop(expr, do: body) do
    quote do
      mlir do
        Beaver.MLIR.Dialect.SCF.while [] do
          region do
            block _() do
              condition = unquote(expr)
              Beaver.MLIR.Dialect.SCF.condition(condition) >>> []
            end
          end

          region do
            block _() do
              unquote(body)
              Beaver.MLIR.Dialect.SCF.yield() >>> []
            end
          end
        end >>> []
      end
    end
  end

  defmacro cond_br(condition, clauses) do
    true_body = Keyword.fetch!(clauses, :do)
    false_body = Keyword.fetch!(clauses, :else)

    quote do
      mlir do
        CF.cond_br(
          unquote(condition),
          block do
            unquote(true_body)
          end,
          block do
            unquote(false_body)
          end,
          loc: Beaver.MLIR.Location.from_env(unquote(Macro.escape(__CALLER__)))
        ) >>> []
      end
    end
  end

  defmacro struct_if(condition, clauses) do
    true_body = Keyword.fetch!(clauses, :do)
    false_body = clauses[:else]

    quote do
      mlir do
        alias Beaver.MLIR.Dialect.SCF

        SCF.if [unquote(condition)] do
          region do
            block _true() do
              unquote(true_body)
              SCF.yield() >>> []
            end
          end

          region do
            block _false() do
              unquote(false_body)
              SCF.yield() >>> []
            end
          end
        end >>> []
      end
    end
  end

  defp normalize_call(call) do
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
          {a = {name, _, nil}, _} when is_atom(name) ->
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
    {call, ret_types} =
      case call do
        {:"::", _, [call, ret_type]} -> {call, [ret_type]}
        call -> {call, []}
      end

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
