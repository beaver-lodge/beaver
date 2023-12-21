defmodule TranslateMLIR do
  defmacro __using__(_) do
    quote do
      import TranslateMLIR
    end
  end

  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith, MemRef, Index, SCF}
  alias Beaver.MLIR.Type
  require Func

  defp gen_mlir(
         {:for, _,
          [
            {:<-, _, [{loop_arg, _, nil}, list]},
            [
              do: do_block
            ]
          ]},
         acc,
         ctx,
         block
       )
       when is_list(list) do
    if list |> Enum.all?(&is_integer/1) do
      {values, acc} = Macro.prewalk(list, acc, &gen_mlir(&1, &2, ctx, block))

      mlir ctx: ctx, block: block do
        # generate result to write to
        result = MemRef.alloc() >>> MLIR.Type.memref([length(list)], MLIR.Type.i64())
        # generate list literal
        memref = MemRef.alloca() >>> MLIR.Type.memref([length(list)], MLIR.Type.i64())

        for {v, i} <- Enum.with_index(values) do
          indices = Index.constant(value: Attribute.index(i)) >>> Type.index()
          MemRef.store(v, memref, indices) >>> []
        end

        # generate for loop
        lower_bound = Index.constant(value: Attribute.index(0)) >>> Type.index()
        upper_bound = Index.constant(value: Attribute.index(length(list))) >>> Type.index()
        step = Index.constant(value: Attribute.index(1)) >>> Type.index()

        SCF.for [lower_bound, upper_bound, step] do
          region do
            block body(indices >>> Type.index()) do
              arg = MemRef.load(memref, indices) >>> MLIR.Type.i64()
              acc = put_in(acc.variables[loop_arg], arg)

              {ret, _acc} =
                Macro.prewalk(do_block, acc, &gen_mlir(&1, &2, ctx, Beaver.Env.block()))

              MemRef.store(ret, result, indices) >>> []
              SCF.yield() >>> []
            end
          end
        end >>> []
      end

      {result, acc}
    else
      raise "can only compile list of int literal"
    end
  end

  defp gen_mlir(
         {:=, _,
          [
            {name, _, nil},
            bound
          ]},
         acc,
         ctx,
         block
       ) do
    {value, acc} =
      mlir ctx: ctx, block: block do
        Macro.prewalk(bound, acc, &gen_mlir(&1, &2, ctx, Beaver.Env.block()))
      end

    {value, put_in(acc.variables[name], value)}
  end

  defp gen_mlir(
         {:+, _, [left, right]},
         acc,
         ctx,
         block
       ) do
    value =
      mlir ctx: ctx, block: block do
        {[left, right], acc} =
          Enum.reduce([left, right], {[], acc}, fn i, {ops, acc} ->
            {r, acc} = Macro.prewalk(i, acc, &gen_mlir(&1, &2, ctx, Beaver.Env.block()))
            {ops ++ [r], acc}
          end)

        Arith.addi(left, right) >>> MLIR.CAPI.mlirValueGetType(left)
      end

    {value, acc}
  end

  defp gen_mlir(
         {name, [line: _], nil},
         acc,
         _ctx,
         _block
       ) do
    {Map.fetch!(acc.variables, name), acc}
  end

  defp gen_mlir(i, acc, ctx, block) when is_integer(i) do
    value =
      mlir ctx: ctx, block: block do
        Arith.constant(value: Attribute.integer(Type.i64(), i)) >>> Type.i64()
      end

    {value, acc}
  end

  defp gen_mlir(
         ast,
         acc,
         _ctx,
         _block
       ) do
    {ast, acc}
  end

  defp compile_args(args, ctx) do
    arg_type_pairs =
      for {:"::", _, [{a, _, nil}, type]} <- args do
        t =
          case type do
            {:i64, _line, nil} ->
              Type.i64(ctx: ctx)
          end

        {a, t}
      end

    arg_types = Enum.map(arg_type_pairs, &elem(&1, 1))
    entry_block = MLIR.Block.create([])
    args = Beaver.MLIR.Block.add_arg!(entry_block, ctx, arg_types)

    {entry_block, arg_types, args}
  end

  defp compile_body(ctx, block, expr, acc_init) do
    mlir ctx: ctx, block: block do
      {ret, _acc} =
        Macro.prewalk(expr, acc_init, &gen_mlir(&1, &2, ctx, Beaver.Env.block()))

      ret[:do]
    end
  end

  defp compile_defm(call, expr, ctx) do
    {name, args} = Macro.decompose_call(call)

    arg_names =
      for {:"::", _, [{a, _, nil}, _type]} <- args do
        a
      end

    {entry_block, arg_types, args} = compile_args(args, ctx)

    ret_val =
      case compile_body(ctx, entry_block, expr, %{variables: Map.new(Enum.zip(arg_names, args))}) do
        {:__block__, [], values} when is_list(values) -> List.last(values)
        %MLIR.Value{} = v -> v
      end

    mlir ctx: ctx, block: entry_block do
      Func.return(ret_val) >>> []
    end

    mlir ctx: ctx do
      module do
        Func.func main(
                    sym_name: "\"#{name}\"",
                    function_type: Type.function(arg_types, [MLIR.CAPI.mlirValueGetType(ret_val)])
                  ) do
          region do
            MLIR.CAPI.mlirRegionAppendOwnedBlock(Beaver.Env.region(), entry_block)
          end
        end
      end
      |> MLIR.dump!()
    end
  end

  def compile_and_invoke(ir, function, arguments, return) when is_bitstring(ir) do
    ctx = MLIR.Context.create()
    import MLIR.{Transforms, Conversion}

    jit =
      ~m{#{ir}}.(ctx)
      |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
      |> convert_scf_to_cf
      |> convert_arith_to_llvm()
      |> convert_index_to_llvm()
      |> convert_func_to_llvm()
      |> MLIR.Pass.Composer.append("finalize-memref-to-llvm")
      |> reconcile_unrealized_casts
      |> MLIR.Pass.Composer.run!()
      |> MLIR.ExecutionEngine.create!()

    ret =
      MLIR.ExecutionEngine.invoke!(jit, "#{function}", arguments, return)
      |> Beaver.Native.to_term()

    MLIR.ExecutionEngine.destroy(jit)
    MLIR.Context.destroy(ctx)
    ret
  end

  defmacro defm(call, expr \\ nil) do
    {name, args} = Macro.decompose_call(call)
    args = for {:"::", _, [arg, _type]} <- args, do: arg
    ctx = MLIR.Context.create()

    ir =
      compile_defm(call, expr, ctx)
      |> MLIR.Operation.verify!()
      |> MLIR.to_string()

    MLIR.Context.destroy(ctx)

    quote do
      @ir unquote(ir)
      def unquote(name)(unquote_splicing(args)) do
        arguments = [unquote_splicing(args)] |> Enum.map(&Beaver.Native.I64.make/1)
        return = Beaver.Native.I64.make(0)
        TranslateMLIR.compile_and_invoke(@ir, unquote(name), arguments, return)
      end
    end
  end
end
