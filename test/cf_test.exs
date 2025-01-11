defmodule CfTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Type
  alias Beaver.MLIR.Dialect.{Func, CF, Arith}
  require Func
  @moduletag :smoke

  defmodule MutCompiler do
    use Beaver
    require Beaver.MLIR

    defmodule Acc do
      @enforce_keys [:vars, :block, :region]
      defstruct vars: %{}, block: nil, region: nil, ctx: nil
    end

    # 3. use the var in acc
    defp fetch_var(%Acc{vars: vars}, {var_name, _line, nil}) do
      found = Map.get(vars, var_name)

      if is_nil(found) do
        :not_found
      else
        {:ok, found}
      end
    end

    defp put_var(%Acc{vars: vars} = acc, {var_name, _line, nil}, var) do
      %Acc{acc | vars: Map.put(vars, var_name, var)}
    end

    defp update_block(%Acc{block: _old} = acc, block) do
      %Acc{acc | block: block}
    end

    # 1. starts with "root", the return expression
    defp gen_mlir(
           {:return, _line, [arg]},
           %Acc{block: block, ctx: ctx} = acc
         ) do
      # we expect it to be a MLIR Value
      {arg = %MLIR.Value{}, acc} = gen_mlir(arg, acc)

      mlir =
        mlir blk: block, ctx: ctx do
          Func.return(arg) >>> []
        end

      {mlir, acc}
    end

    # found {:base_lr, [line: 89], nil} unmatched, so we add this match to extract MLIR Value
    defp gen_mlir({_var_name, _line, nil} = ast, acc) do
      # {ast, acc}
      case fetch_var(acc, ast) do
        {:ok, found} ->
          {found, acc}

        :not_found ->
          raise "block arg not found, #{inspect(ast)}"
          {gen_mlir(ast, acc), acc}
      end
    end

    # For `:if` it is kind of tricky, we need to generate block for it
    defp gen_mlir(
           {:if, _, [cond_ast, [do: do_block_ast, else: else_block_ast]]},
           %Acc{region: region, block: entry, ctx: ctx} = acc
         ) do
      {condition, acc} = gen_mlir(cond_ast, acc)

      bb_next =
        mlir ctx: ctx do
          block _(arg >>> Type.f32()) do
          end
        end

      true_branch =
        mlir ctx: ctx do
          block do
            {mlir, acc} = gen_mlir(do_block_ast, update_block(acc, Beaver.Env.block()))

            CF.br({bb_next, [mlir]}) >>> []
          end
        end

      false_branch =
        mlir ctx: ctx do
          block do
            {mlir, acc} =
              gen_mlir(else_block_ast, update_block(acc, Beaver.Env.block()))

            CF.br({bb_next, [mlir]}) >>> []
          end
        end

      mlir blk: entry, ctx: ctx do
        CF.cond_br(condition, true_branch, false_branch) >>> []
      end

      Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, true_branch)
      Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, false_branch)
      Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, bb_next)

      {arg, update_block(acc, bb_next)}
    end

    # an assign, it is different from binding in Elixir, so we want to generate IR of mutable semantic
    # but this could be complicated, we can simply call gen_mlir with the ast of the var for now
    defp gen_mlir(
           {:=, _,
            [
              {_name, _, nil} = ast,
              var
            ]},
           acc
         ) do
      {mlir, acc} = gen_mlir(var, acc)
      {mlir, put_var(acc, ast, mlir)}
    end

    # in real world, this should be merged with the gen_mlir for :+
    defp gen_mlir({:<, _, [left, right]}, %Acc{block: block, ctx: ctx} = acc) do
      {left = %MLIR.Value{}, acc} = gen_mlir(left, acc)
      {right = %MLIR.Value{}, acc} = gen_mlir(right, acc)

      less =
        mlir blk: block, ctx: ctx do
          Arith.cmpf(left, right, predicate: Arith.cmp_f_predicate(:ugt)) >>> Type.i1()
        end

      {less, acc}
    end

    # after adding this, you should see this kind of IR printed
    # %0 = arith.mulf %arg2, %arg1 : f32
    defp gen_mlir({:*, _line, [left, right]}, %Acc{block: block, ctx: ctx} = acc) do
      {left = %MLIR.Value{}, acc} = gen_mlir(left, acc)
      {right = %MLIR.Value{}, acc} = gen_mlir(right, acc)
      # we only work with float 32 for now
      add =
        mlir blk: block, ctx: ctx do
          Arith.mulf(left, right) >>> Type.f32()
        end

      {add, acc}
    end

    # at some point you should see logging of node not supported for MLIR CAPI Value,
    # let's add the match to disable this kind of logging
    defp gen_mlir(%MLIR.Value{} = mlir, acc) do
      {mlir, acc}
    end

    # you might want to print the node not matched
    defp gen_mlir(ast, acc) do
      # IO.inspect(ast, label: "node not matched")
      {ast, acc}
    end

    def gen_func(call, block) do
      {name, _args} = Macro.decompose_call(call)

      ctx = MLIR.Context.create()

      mlir ctx: ctx do
        module do
          Func.func some_func(
                      sym_name: "\"#{name}\"",
                      function_type: Type.function(List.duplicate(Type.f(32), 4), [Type.f(32)])
                    ) do
            region do
              block _bb_entry(
                      total_iters >>> Type.f32(),
                      factor >>> Type.f(32),
                      base_lr >>> Type.f(32),
                      step >>> Type.f(32)
                    ) do
                # Put the MLIR Values for args into a Map
                vars = %{total_iters: total_iters, factor: factor, base_lr: base_lr, step: step}

                acc = %Acc{
                  vars: vars,
                  region: Beaver.Env.region(),
                  block: Beaver.Env.block(),
                  ctx: Beaver.Env.context()
                }

                # keep generating until we meet a terminator
                {_mlir, _acc} =
                  Macro.prewalk(block, acc, fn ast, %Acc{} = acc ->
                    gen_mlir(ast, acc)
                  end)
              end
            end
          end
        end
      end
      # we let MLIR verify the generated IR for us, so it gonna be legit!
      |> MLIR.verify!()
    end

    # In most of LLVM or other compiler guidance, it starts with ast parsing.
    # In Elixir we don't have to, just reuse the Elixir ast and use macro to do the magic!
    defmacro defnative(call, do: block) do
      mlir_asm =
        MutCompiler.gen_func(call, block)
        |> MLIR.to_string()

      quote do
        alias MLIR.Dialect.Func
        unquote(Macro.escape(mlir_asm))
      end
    end

    defp lower(ir) do
      import MLIR.{Transform, Conversion}

      ir
      |> convert_cf_to_llvm()
      |> convert_arith_to_llvm
      |> Beaver.Composer.nested("func.func", "llvm-request-c-wrappers")
      |> Beaver.Composer.nested("func.func", {"DoNothing0", "func.func", fn _, _ -> :ok end})
      |> convert_func_to_llvm
      |> canonicalize
      |> Beaver.Composer.append({"DoNothing1", "builtin.module", fn _, _ -> :ok end})
      |> Beaver.Composer.run!()
    end

    def get_func(ir, func) do
      llvm_ir = lower(ir)

      jit = MLIR.ExecutionEngine.create!(llvm_ir)

      fn total_iters, factor, base_lr, step ->
        return = Beaver.Native.F32.make(0.9999)

        arguments =
          [
            total_iters,
            factor,
            base_lr,
            step
          ]
          |> Enum.map(&Beaver.Native.F32.make/1)

        MLIR.ExecutionEngine.invoke!(jit, func, arguments, return)
        |> Beaver.Native.to_term()
      end
    end
  end

  test "cf with mutation", %{ctx: ctx} do
    import MutCompiler

    mlir =
      defnative get_lr(total_iters, factor, base_lr, step) do
        base_lr = base_lr * factor

        return(base_lr)
      end

    assert mlir =~ "%0 = arith.mulf %arg2, %arg1 : f32", mlir

    mlir =
      defnative get_lr_with_ctrl_flow(total_iters, factor, base_lr, step) do
        base_lr =
          if step < total_iters do
            base_lr * factor
          else
            base_lr
          end

        return(base_lr)
      end

    ir = ~m{#{mlir}}.(ctx)

    f = get_func(ir, "get_lr_with_ctrl_flow")

    assert f.(1000.0, 0.5, 0.002, 200.0) /
             f.(1000.0, 0.5, 0.002, 2000.0) == 2

    assert mlir =~ "%1 = arith.mulf %arg2, %arg1 : f32", mlir
    assert mlir =~ "return %2 : f32", mlir
  end
end
