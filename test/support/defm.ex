defmodule TranslateMLIR do
  defmacro __using__(_) do
    quote do
      import TranslateMLIR
    end
  end

  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith}
  alias Beaver.MLIR.Type
  require Func

  defp gen_mlir(
         {:some_llvm_add_mlir_operation, _, [left, right]} = ast,
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

        Arith.addi(left, right) >>> Type.i64()
      end

    {value, acc}
  end

  defp gen_mlir(
         {name, [line: _], nil},
         acc,
         ctx,
         block
       ) do
    {acc.variables[name], acc}
  end

  defp gen_mlir(
         ast,
         acc,
         ctx,
         block
       ) do
    {ast, acc}
  end

  def compile_defm(call, expr, ctx) do
    {name, args} = Macro.decompose_call(call)

    mlir ctx: ctx do
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

      body_block = MLIR.Block.create([])

      ret_type =
        mlir ctx: ctx, block: body_block do
          Beaver.MLIR.Block.add_arg!(
            Beaver.Env.block(),
            Beaver.Env.context(),
            arg_types
          )

          variables =
            for {{k, _}, i} <- Enum.with_index(arg_type_pairs) do
              {k, Beaver.MLIR.Block.get_arg!(Beaver.Env.block(), i)}
            end

          acc_init = %{variables: Map.new(variables)}

          {ret, _acc} =
            Macro.prewalk(expr, acc_init, &gen_mlir(&1, &2, ctx, Beaver.Env.block()))

          ret_val = ret[:do]
          Func.return(ret_val) >>> []
          MLIR.CAPI.mlirValueGetType(ret_val)
        end

      module do
        Func.func main(
                    sym_name: "\"#{name}\"",
                    function_type: Type.function(arg_types, [ret_type])
                  ) do
          region do
            MLIR.CAPI.mlirRegionAppendOwnedBlock(Beaver.Env.region(), body_block)
          end
        end
      end
    end
  end

  defmacro defm(call, expr \\ nil) do
    {name, args} = Macro.decompose_call(call)
    args = for {:"::", _, [arg, type]} <- args, do: arg
    ctx = MLIR.Context.create()

    ir =
      compile_defm(call, expr, ctx)
      |> MLIR.dump!()
      |> MLIR.Operation.verify!()
      |> MLIR.to_string()

    ctx = MLIR.Context.destroy(ctx)

    quote do
      @ir unquote(ir)
      def unquote(name)(unquote_splicing(args)) do
        ctx = MLIR.Context.create()

        arguments = [unquote_splicing(args)] |> Enum.map(&Beaver.Native.I64.make/1)
        return = Beaver.Native.I64.make(0)

        jit =
          ~m{#{@ir}}.(ctx)
          |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
          |> MLIR.Conversion.convert_arith_to_llvm()
          |> MLIR.Conversion.convert_func_to_llvm()
          |> MLIR.Pass.Composer.run!()
          |> MLIR.dump!()
          |> MLIR.ExecutionEngine.create!()

        MLIR.ExecutionEngine.invoke!(jit, "#{unquote(name)}", arguments, return)
        |> Beaver.Native.to_term()
      end
    end
  end
end
