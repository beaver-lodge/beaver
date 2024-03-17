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
      require Beaver.MIF.{BEAM, Pointer}
      require Beaver.MLIR.Dialect.Func
      alias Beaver.MLIR.Dialect.{Func, Arith, LLVM, CF}
      alias MLIR.{Type, Attribute}
      import Beaver.MLIR.Type

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

  @doc """
  getting the function type and entry block argument types of an mif, and the index of beam env
  """
  def mif_type(arg_num) do
    term_t = Beaver.ENIF.mlir_t(:term)
    arg_types = List.duplicate(term_t, arg_num) ++ [Beaver.ENIF.mlir_t(:env)]
    {Type.function(arg_types, [term_t]), arg_types, arg_num}
  end

  defp inject_mlir_opts({:^, _, [{_, _, _} = var]}) do
    quote do
      CF.br({unquote(var), []}) >>> []
    end
  end

  @enif_functions Beaver.ENIF.functions()
  @intrinsics @enif_functions ++
                [
                  :result_at,
                  :!=
                ]

  defp inject_mlir_opts({f, _, args}) when f in @intrinsics do
    quote do
      Beaver.MIF.handle_intrinsic(unquote(f), [unquote_splicing(args)],
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block()
      )
    end
  end

  defp inject_mlir_opts(ast) do
    with {{:__aliases__, _, _} = m, f, args} <- Macro.decompose_call(ast) do
      quote do
        unquote(m).handle_intrinsic(unquote(f), [unquote_splicing(args)],
          ctx: Beaver.Env.context(),
          block: Beaver.Env.block()
        )
      end
    else
      :error ->
        ast

      _ ->
        ast
    end
  end

  defp definition_to_func({_env, call, body}) do
    {name, args} = Macro.decompose_call(call)

    body =
      body[:do]
      |> Macro.postwalk(&inject_mlir_opts(&1))
      |> List.wrap()

    quote do
      mlir ctx: var!(ctx), block: var!(block) do
        args_num = unquote(length(args))
        {function_type, arg_types, env_index} = Beaver.MIF.mif_type(args_num)

        Beaver.MLIR.Dialect.Func.func unquote(name)(function_type: function_type) do
          region do
            block _entry() do
              MLIR.Block.add_args!(Beaver.Env.block(), arg_types, ctx: Beaver.Env.context())
              var!(mif_internal_beam_env) = MLIR.Block.get_arg!(Beaver.Env.block(), env_index)

              [
                unquote_splicing(
                  for arg <- args do
                    quote do
                      Kernel.var!(unquote(arg))
                    end
                  end
                )
              ] =
                Range.new(0, args_num - 1)
                |> Enum.map(&MLIR.Block.get_arg!(Beaver.Env.block(), &1))

              unquote(body)
            end
          end
        end
      end
    end
  end

  def compile_definitions(definitions) do
    ctx = Beaver.MLIR.Context.create()

    m =
      mlir ctx: ctx do
        module do
          Beaver.ENIF.populate_external_functions(ctx, Beaver.Env.block())

          for {env, _, _} = d <- definitions do
            f = definition_to_func(d)
            binding = [ctx: Beaver.Env.context(), block: Beaver.Env.block()]
            Code.eval_quoted(f, binding, env)
          end
        end
      end
      |> MLIR.Operation.verify!(debug: true)
      |> MLIR.dump!()
      |> MLIR.to_string()

    MLIR.Context.destroy(ctx)
    m
  end

  defmacro op({:"::", _, [call, types]}, _ \\ []) do
    {{:., _, [{dialect, meta, nil}, op]}, _, args} = call

    quote do
      %Beaver.SSA{
        op: unquote("#{dialect}.#{op}"),
        arguments: unquote(args),
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block(),
        loc: Beaver.MLIR.Location.file(name: "from_quoted", line: unquote(meta[:line]))
      }
      |> Beaver.SSA.put_results([unquote_splicing(List.wrap(types))])
      |> MLIR.Operation.create()
    end
  end

  defp constant_of_same_type(i, v, opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      t = MLIR.CAPI.mlirValueGetType(v)
      Arith.constant(value: Attribute.integer(t, i)) >>> t
    end
  end

  defp wrap_arg({i, t}, opts) when is_integer(i) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      case i do
        %MLIR.Value{} ->
          i

        i when is_integer(i) ->
          Arith.constant(value: Attribute.integer(t, i)) >>> t
      end
    end
  end

  defp wrap_arg({v, _}, _) do
    v
  end

  def handle_intrinsic(:result_at, [%MLIR.Value{} = v, i], _opts) when is_integer(i) do
    v
  end

  def handle_intrinsic(:result_at, [l, i], _opts) when is_list(l) do
    l |> Enum.at(i)
  end

  def handle_intrinsic(:result_at, [%MLIR.Operation{} = op, i], _opts) do
    MLIR.CAPI.mlirOperationGetResult(op, i)
  end

  def handle_intrinsic(:!=, [left, right], opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      [left, right] =
        case {left, right} do
          {%MLIR.Value{} = v, i} when is_integer(i) ->
            [v, constant_of_same_type(i, v, opts)]

          {i, %MLIR.Value{} = v} when is_integer(i) ->
            [constant_of_same_type(i, v, opts), v]

          {%MLIR.Value{}, %MLIR.Value{}} ->
            [left, right]
        end

      Arith.cmpi(left, right, predicate: Arith.cmp_i_predicate(:ne)) >>> Type.i1()
    end
  end

  def handle_intrinsic(name, args, opts) when name in @enif_functions do
    {arg_types, ret_types} = Beaver.ENIF.signature(opts[:ctx], name)
    args = args |> Enum.zip(arg_types) |> Enum.map(&wrap_arg(&1, opts))

    mlir ctx: opts[:ctx], block: opts[:block] do
      Func.call(args, callee: Attribute.flat_symbol_ref("#{name}")) >>>
        case ret_types do
          [ret] ->
            ret

          [] ->
            []
        end
    end
  end

  defmacro cond_br(condition, clauses) do
    true_body = Keyword.fetch!(clauses, :do)
    false_body = Keyword.fetch!(clauses, :else)

    quote do
      use Beaver

      mlir do
        CF.cond_br(
          unquote(condition),
          block do
            unquote(true_body)
          end,
          block do
            unquote(false_body)
          end
        ) >>> []
      end
    end
  end

  defmacro defm(call, body \\ []) do
    {name, args} = Macro.decompose_call(call)
    env = __CALLER__

    quote do
      @defm unquote(Macro.escape({env, call, body}))
      def unquote(name)(unquote_splicing(args)) do
        %{jit: jit} = Agent.get(__MODULE__, & &1)

        Beaver.MLIR.CAPI.mif_raw_jit_invoke_with_terms(
          jit.ref,
          to_string(unquote(name)),
          unquote(args)
        )
      end
    end
  end

  def init_jit(module) do
    import Beaver.MLIR.Conversion
    ctx = MLIR.Context.create()
    Beaver.Diagnostic.attach(ctx)

    jit =
      ~m{#{module.__ir__()}}.(ctx)
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

    Agent.start_link(fn -> %{ctx: ctx, jit: jit} end, name: module)
  end

  def destroy_jit(module) do
    %{ctx: ctx, jit: jit} = Agent.get(module, & &1)
    MLIR.ExecutionEngine.destroy(jit)
    MLIR.Context.destroy(ctx)
    Agent.stop(module)
  end
end
