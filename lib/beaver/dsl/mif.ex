defmodule Beaver.MIF do
  @doc """
  A MIF is NIF generated by LLVM/MLIR
  """
  require Beaver.Env
  alias Beaver.MLIR.{Type, Attribute}
  alias Beaver.MLIR.Dialect.{Arith, LLVM, Func}
  use Beaver

  defmacro __using__(_opts) do
    quote do
      import Beaver.MIF
      @before_compile Beaver.MIF
      Module.register_attribute(__MODULE__, :defm, accumulate: true)
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      {ir_str, _} = @defm |> Beaver.MIF.compile_definitions(__ENV__) |> Code.eval_quoted()
      @ir ir_str
      def __ir__ do
        @ir
      end
    end
  end

  defp getter_symbol(ctx, t) do
    cond do
      Type.equal?(t, Type.i32(ctx: ctx)) -> "enif_get_int"
      Type.equal?(t, Type.i64(ctx: ctx)) -> "enif_get_int64"
      true -> raise "getter symbol not found"
    end
    |> Attribute.flat_symbol_ref()
  end

  defp maker_symbol(ctx, t) do
    cond do
      Type.equal?(t, Type.i32(ctx: ctx)) -> "enif_make_int"
      Type.equal?(t, Type.i64(ctx: ctx)) -> "enif_make_int64"
      true -> raise "maker symbol found"
    end
    |> Attribute.flat_symbol_ref()
  end

  def arg_from_term(ctx, block, env, type, index) do
    mlir ctx: ctx, block: block do
      one = Arith.constant(value: Attribute.integer(Type.i(32), 1)) >>> ~t<i32>
      ptr = LLVM.alloca(one, elem_type: type) >>> ~t{!llvm.ptr}
      term = MLIR.Block.get_arg!(Beaver.Env.block(), index)
      Func.call([env, term, ptr], callee: getter_symbol(ctx, type)) >>> Type.i32()
      LLVM.load(ptr) >>> type
    end
  end

  def value_to_term(ctx, block, env, value) do
    mlir ctx: ctx, block: block do
      Func.call([env, value], callee: maker_symbol(ctx, MLIR.CAPI.mlirValueGetType(value))) >>>
        Beaver.ENIF.ERL_NIF_TERM.mlir_t()
    end
  end

  def compile_definitions(definitions, env) do
    functions =
      for {call, expr} <- definitions do
        {name, args} = Macro.decompose_call(call)

        expr = Macro.postwalk(expr, &Macro.expand(&1, env))[:do] |> List.wrap()

        quote do
          arg_types =
            List.duplicate(Beaver.ENIF.ERL_NIF_TERM.mlir_t(), unquote(length(args))) ++
              [
                Beaver.ENIF.ErlNifEnv.mlir_t()
              ]

          ret_types = [Beaver.ENIF.ERL_NIF_TERM.mlir_t()]

          Beaver.MLIR.Dialect.Func.func unquote(name)(
                                          function_type: Type.function(arg_types, ret_types)
                                        ) do
            region do
              block _() do
                MLIR.Block.add_args!(Beaver.Env.block(), arg_types, ctx: Beaver.Env.context())
                env = MLIR.Block.get_arg!(Beaver.Env.block(), length(arg_types) - 1)

                unquote(
                  for {{:"::", _line0, [{_arg, _line1, nil} = var, t]}, index} <-
                        Enum.with_index(args) do
                    quote do
                      Kernel.var!(unquote(var)) =
                        Beaver.MIF.arg_from_term(
                          Beaver.Env.context(),
                          Beaver.Env.block(),
                          env,
                          unquote(t),
                          unquote(index)
                        )
                    end
                  end
                )

                last_op = unquote_splicing(expr)
                ret = last_op |> Beaver.Walker.results() |> Enum.at(0)
                ret_term = Beaver.MIF.value_to_term(ctx, Beaver.Env.block(), env, ret)
                Beaver.MLIR.Dialect.Func.return(ret_term) >>> []
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
            alias Beaver.MLIR.Dialect.{Func, Arith, LLVM}
            alias MLIR.{Type, Attribute}
            import Beaver.MLIR.Type
            Beaver.ENIF.populate_external_functions(ctx, Beaver.Env.block())

            (unquote_splicing(functions))
          end
        end
        |> MLIR.to_string()

      MLIR.Context.destroy(ctx)
      m
    end
  end

  defmacro op({:"::", _, [call, types]}, _ \\ []) do
    {{:., _, [{dialect, _, nil}, op]}, _, args} = call

    quote do
      %Beaver.SSA{
        op: unquote("#{dialect}.#{op}"),
        arguments: unquote(args),
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block(),
        loc: Beaver.MLIR.Location.from_env(__ENV__)
      }
      |> Beaver.SSA.put_results([unquote_splicing(List.wrap(types))])
      |> MLIR.Operation.create()
    end
  end

  defmacro defm(call, expr \\ []) do
    {name, args} = Macro.decompose_call(call)
    args = for {:"::", _, [arg, _type]} <- args, do: arg

    quote do
      @defm unquote(Macro.escape({call, expr}))
      def unquote(name)(unquote_splicing(args)) do
        import Beaver.MLIR.Conversion
        arguments = [unquote_splicing(args)]
        ctx = MLIR.Context.create()
        Beaver.Diagnostic.attach(ctx)

        jit =
          ~m{#{__ir__()}}.(ctx)
          |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
          |> convert_scf_to_cf
          |> convert_arith_to_llvm()
          |> convert_index_to_llvm()
          |> convert_func_to_llvm()
          |> MLIR.Pass.Composer.append("finalize-memref-to-llvm")
          |> reconcile_unrealized_casts
          |> MLIR.Pass.Composer.run!(print: System.get_env("DEFM_PRINT_IR") == "1")
          |> MLIR.ExecutionEngine.create!()

        :ok = Beaver.MLIR.CAPI.mif_raw_jit_register_enif(jit.ref)

        Beaver.MLIR.CAPI.mif_raw_jit_invoke_with_terms(
          jit.ref,
          to_string(unquote(name)),
          arguments
        )
      end
    end
  end
end
