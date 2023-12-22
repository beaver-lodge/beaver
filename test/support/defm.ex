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

  defp compile_for_loop(result, write_index, [[do: do_block]], acc, ctx, block) do
    mlir ctx: ctx, block: block do
      {ret, acc} = Macro.prewalk(do_block, acc, &gen_mlir(&1, &2, ctx, block))
      MemRef.store(ret, result, write_index) >>> []
      {ret, acc}
    end
  end

  defp compile_for_loop(
         result,
         write_index,
         [{:<-, _, [{loop_arg, _, nil}, memref]} | tail],
         acc,
         ctx,
         block
       ) do
    mlir ctx: ctx, block: block do
      zero = Index.constant(value: Attribute.index(0)) >>> Type.index()
      lower_bound = Index.constant(value: Attribute.index(0)) >>> Type.index()
      upper_bound = MemRef.dim(memref, zero) >>> Type.index()
      step = Index.constant(value: Attribute.index(1)) >>> Type.index()

      SCF.for [lower_bound, upper_bound, step] do
        region do
          block body(indices >>> Type.index()) do
            arg = MemRef.load(memref, indices) >>> MLIR.Type.i64()
            acc = put_in(acc.variables[loop_arg], arg)
            write_index = Index.mul(write_index, upper_bound) >>> Type.index()
            write_index = Index.add(write_index, indices) >>> Type.index()
            compile_for_loop(result, write_index, tail, acc, ctx, Beaver.Env.block())
            SCF.yield() >>> []
          end
        end
      end >>> []
    end

    {result, acc}
  end

  defp gen_mlir(
         {:for, _, expressions},
         acc,
         ctx,
         block
       ) do
    mlir ctx: ctx, block: block do
      {expressions, acc} =
        expressions
        |> Enum.reduce({[], acc}, fn
          {:<-, line1, [{loop_arg, line2, nil}, enum]}, {new_expressions, acc} ->
            {memref, acc} = Macro.prewalk(enum, acc, &gen_mlir(&1, &2, ctx, block))
            {new_expressions ++ [{:<-, line1, [{loop_arg, line2, nil}, memref]}], acc}

          expr, {new_expressions, acc} ->
            {new_expressions ++ [expr], acc}
        end)

      zero = Index.constant(value: Attribute.index(0)) >>> Type.index()
      one = Index.constant(value: Attribute.index(1)) >>> Type.index()

      size =
        Enum.reduce(expressions, one, fn
          {:<-, _line1, [{_loop_arg, _line2, nil}, memref]}, sum ->
            size = MemRef.dim(memref, zero) >>> Type.index()
            Index.mul(size, sum) >>> Type.index()

          _, sum ->
            sum
        end)

      result =
        MemRef.alloc(size, operand_segment_sizes: Beaver.MLIR.ODS.operand_segment_sizes([1, 0])) >>>
          MLIR.Type.memref([:dynamic], MLIR.Type.i64())

      compile_for_loop(result, zero, expressions, acc, ctx, block)
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
         [head | _tail] = list,
         acc,
         ctx,
         block
       )
       when is_list(list) and is_integer(head) do
    if list |> Enum.all?(&is_integer/1) do
      {values, acc} =
        Enum.reduce(list, {[], acc}, fn i, {ops, acc} ->
          {r, acc} = Macro.prewalk(i, acc, &gen_mlir(&1, &2, ctx, block))
          {ops ++ [r], acc}
        end)

      mlir ctx: ctx, block: block do
        memref = MemRef.alloca() >>> MLIR.Type.memref([length(list)], MLIR.Type.i64())

        for {v, i} <- Enum.with_index(values) do
          indices = Index.constant(value: Attribute.index(i)) >>> Type.index()
          MemRef.store(v, memref, indices) >>> []
        end
      end

      {memref, acc}
    else
      raise "can only compile list of int literal. got: \n#{inspect(list, pretty: true)}"
    end
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

  def memref_descriptor_ptr() do
    Beaver.Native.Memory.new(nil, sizes: [-1], type: {:s, 64})
  end

  def parse_integers(return) do
    [len] = return.descriptor |> Beaver.Native.Memory.Descriptor.sizes()

    return.descriptor
    |> Beaver.Native.Memory.Descriptor.aligned()
    |> Beaver.Native.OpaquePtr.to_binary(8 * len)
    |> do_parse_integers([])
  end

  defp do_parse_integers(<<>>, integers), do: Enum.reverse(integers)

  defp do_parse_integers(<<int::little-integer-size(64), rest::binary>>, integers) do
    do_parse_integers(rest, [int | integers])
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

    ret_t = MLIR.CAPI.mlirValueGetType(ret_val)

    return_maker =
      if MLIR.CAPI.mlirTypeIsAInteger(ret_t) |> Beaver.Native.to_term() do
        %{mode: :value, maker: {Beaver.Native.I64, :make, [0]}}
      else
        if MLIR.CAPI.mlirTypeIsAMemRef(ret_t) |> Beaver.Native.to_term() do
          %{
            mode: :mutation,
            maker: {__MODULE__, :memref_descriptor_ptr, []},
            preparer: {Beaver.Native.Memory, :descriptor_ptr},
            postprocesser: {__MODULE__, :parse_integers}
          }
        else
          raise "ret type not supported"
        end
      end

    ir =
      mlir ctx: ctx do
        module do
          Func.func main(
                      sym_name: "\"#{name}\"",
                      function_type:
                        Type.function(arg_types, [MLIR.CAPI.mlirValueGetType(ret_val)])
                    ) do
            region do
              MLIR.CAPI.mlirRegionAppendOwnedBlock(Beaver.Env.region(), entry_block)
            end
          end
        end
        |> tap(fn ir ->
          if System.get_env("DEFM_DUMP_IR") == "1" do
            MLIR.dump!(ir)
          end
        end)
      end

    {ir, return_maker}
  end

  def compile_and_invoke(ir, function, arguments, return \\ nil) when is_bitstring(ir) do
    ctx = MLIR.Context.create()
    import MLIR.Conversion

    jit =
      ~m{#{ir}}.(ctx)
      |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
      |> convert_scf_to_cf
      |> convert_arith_to_llvm()
      |> convert_index_to_llvm()
      |> convert_func_to_llvm()
      |> MLIR.Pass.Composer.append("finalize-memref-to-llvm")
      |> reconcile_unrealized_casts
      |> MLIR.Pass.Composer.run!(print: System.get_env("DEFM_PRINT_IR") == "1")
      |> MLIR.ExecutionEngine.create!()

    ret =
      MLIR.ExecutionEngine.invoke!(jit, "#{function}", arguments, return)

    MLIR.ExecutionEngine.destroy(jit)
    MLIR.Context.destroy(ctx)
    ret
  end

  defmacro defm(call, expr \\ nil) do
    {name, args} = Macro.decompose_call(call)
    args = for {:"::", _, [arg, _type]} <- args, do: arg
    ctx = MLIR.Context.create()

    {ir, ret_maker} =
      compile_defm(call, expr, ctx)

    ir = ir |> MLIR.Operation.verify!() |> MLIR.to_string()

    MLIR.Context.destroy(ctx)

    quote do
      @ir unquote(ir)
      def unquote(String.to_atom("__original__#{name}"))(unquote_splicing(args)) do
        unquote(expr[:do])
      end

      def unquote(name)(unquote_splicing(args)) do
        arguments = [unquote_splicing(args)] |> Enum.map(&Beaver.Native.I64.make/1)

        %{mode: return_mode, maker: {mod, func, args}} =
          maker =
          unquote(Macro.escape(ret_maker))

        return = apply(mod, func, args)

        return_arg =
          if preparer = maker[:preparer] do
            {mod, func} = preparer
            apply(mod, func, [return])
          else
            return
          end

        case return_mode do
          :value ->
            TranslateMLIR.compile_and_invoke(@ir, unquote(name), arguments, return_arg)
            |> Beaver.Native.to_term()

          :mutation ->
            TranslateMLIR.compile_and_invoke(@ir, unquote(name), [return_arg | arguments])

            if postprocesser = maker[:postprocesser] do
              {mod, func} = postprocesser
              apply(mod, func, [return])
            else
              return
            end
        end
      end
    end
  end
end
