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

  @enif_functions Beaver.ENIF.functions()
  @intrinsics @enif_functions ++ [:result_at, :!=, :-, :+, :<, :>, :<=, :>=, :==]

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

  defp definition_to_func({call, ret_types, body}) do
    {name, args} = Macro.decompose_call(call)

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
        |> MLIR.dump!()
      end
    end
  end

  def compile_definitions(definitions) do
    ctx = Beaver.MLIR.Context.create()

    m =
      mlir ctx: ctx do
        module do
          Beaver.ENIF.populate_external_functions(ctx, Beaver.Env.block())

          for {env, d} <- definitions do
            f = definition_to_func(d)
            binding = [ctx: Beaver.Env.context(), block: Beaver.Env.block()]
            Code.eval_quoted(f, binding, env)
          end
        end
      end
      |> MLIR.Operation.verify!(debug: false)
      |> MLIR.to_string()

    MLIR.Context.destroy(ctx)
    m
  end

  defmacro op({:"::", _, [call, types]}, _ \\ []) do
    {{:., _, [{dialect, _meta, nil}, op]}, _, args} = call

    quote do
      %Beaver.SSA{
        op: unquote("#{dialect}.#{op}"),
        arguments: unquote(args),
        ctx: Beaver.Env.context(),
        block: Beaver.Env.block(),
        loc: Beaver.MLIR.Location.from_env(unquote(Macro.escape(__CALLER__)))
      }
      |> Beaver.SSA.put_results([unquote_splicing(List.wrap(types))])
      |> MLIR.Operation.create()
    end
  end

  defmacro call({:"::", _, [call, types]}, _ \\ []) do
    {name, args} = Macro.decompose_call(call)

    quote do
      %Beaver.SSA{
        op: unquote("func.call"),
        arguments: [unquote_splicing(args), callee: Attribute.flat_symbol_ref("#{unquote(name)}")],
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

  def handle_intrinsic(op, [left, right], opts) when op in [:!=, :-, :+, :<, :>, :<=, :>=, :==] do
    mlir ctx: opts[:ctx], block: opts[:block] do
      operands =
        case {left, right} do
          {%MLIR.Value{} = v, i} when is_integer(i) ->
            [v, constant_of_same_type(i, v, opts)]

          {i, %MLIR.Value{} = v} when is_integer(i) ->
            [constant_of_same_type(i, v, opts), v]

          {%MLIR.Value{}, %MLIR.Value{}} ->
            [left, right]
        end

      case op do
        :!= ->
          Arith.cmpi(operands, predicate: Arith.cmp_i_predicate(:ne)) >>> Type.i1()

        :== ->
          Arith.cmpi(operands, predicate: Arith.cmp_i_predicate(:eq)) >>> Type.i1()

        :> ->
          Arith.cmpi(operands, predicate: Arith.cmp_i_predicate(:sgt)) >>> Type.i1()

        :>= ->
          Arith.cmpi(operands, predicate: Arith.cmp_i_predicate(:sge)) >>> Type.i1()

        :< ->
          Arith.cmpi(operands, predicate: Arith.cmp_i_predicate(:slt)) >>> Type.i1()

        :<= ->
          Arith.cmpi(operands, predicate: Arith.cmp_i_predicate(:sle)) >>> Type.i1()

        :- ->
          Arith.subi(operands) >>> MLIR.CAPI.mlirValueGetType(left)

        :+ ->
          Arith.addi(operands) >>> MLIR.CAPI.mlirValueGetType(left)
      end
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
    [_env | invoke_args] = args

    invoke_args =
      for {:"::", _, [a, _t]} <- invoke_args do
        a
      end

    quote do
      @defm unquote(Macro.escape({env, {call, ret_types, body}}))
      def unquote(name)(unquote_splicing(invoke_args)) do
        %{jit: jit} = Agent.get(__MODULE__, & &1)

        Beaver.MLIR.CAPI.mif_raw_jit_invoke_with_terms(
          jit.ref,
          to_string(unquote(name)),
          unquote(invoke_args)
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
