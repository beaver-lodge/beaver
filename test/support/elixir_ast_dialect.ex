defmodule ElixirAST do
  require Logger
  @moduledoc "Prototype the Elixir AST dialect."
  use Beaver.Slang, name: "ex"
  deftype dyn
  deftype bound
  deftype unbound
  defop lit_int(), do: [Type.i64()]
  defop var(), do: [any()]
  defop bind(any(), any()), do: [any()]
  defop add(any(), any()), do: [any()]
  alias Beaver.MLIR.Type
  alias Beaver.MLIR.Attribute
  alias Beaver.MLIR.Dialect.Func
  use Beaver

  defp gen_mlir(
         {:defmodule, _,
          [
            {:__aliases__, [alias: false], [name]},
            [
              do: do_body
            ]
          ]},
         ctx,
         block
       ) do
    mlir ctx: ctx, blk: block do
      m =
        module sym_name: ~a{"#{name}"} do
          Macro.prewalk(do_body, &gen_mlir(&1, ctx, Beaver.Env.block()))
        end

      MLIR.CAPI.mlirBlockAppendOwnedOperation(block, MLIR.Operation.from_module(m))
    end
  end

  defp gen_mlir(
         {:def, _,
          [
            {name, _, _},
            [
              do: do_body
            ]
          ]},
         ctx,
         block
       ) do
    mlir ctx: ctx, blk: block do
      Func.func sym_name: "\"#{name}\"", function_type: Type.function([], [dyn()]) do
        region do
          block _func_body do
            %MLIR.Value{} =
              ret_value =
              case Macro.prewalk(do_body, &gen_mlir(&1, ctx, Beaver.Env.block())) do
                {:__block__, _, values} ->
                  values |> List.last()

                %MLIR.Value{} = v ->
                  v
              end

            Func.return(ret_value) >>> []
          end
        end
      end >>>
        []
    end
  end

  defp gen_mlir(
         {:=, [],
          [
            {name, [], _},
            bound
          ]},
         ctx,
         block
       ) do
    mlir ctx: ctx, blk: block do
      bound = Macro.prewalk(bound, &gen_mlir(&1, ctx, block))
      var = __MODULE__.var(name: "\"#{name}\"") >>> __MODULE__.unbound()
      __MODULE__.bind(var, bound) >>> __MODULE__.bound()
    end
  end

  defp gen_mlir(
         {:+, _, [left, right]},
         ctx,
         block
       ) do
    mlir ctx: ctx, blk: block do
      [left, right] =
        for i <- [left, right] do
          Macro.prewalk(i, &gen_mlir(&1, ctx, block))
        end

      __MODULE__.add(left, right) >>> __MODULE__.dyn()
    end
  end

  defp gen_mlir({name, [], mod}, ctx, block) when is_atom(name) and is_atom(mod) do
    mlir ctx: ctx, blk: block do
      __MODULE__.var(name: "\"#{name}\"") >>> __MODULE__.dyn()
    end
  end

  defp gen_mlir({name, [], [] = args}, ctx, block) when is_atom(name) do
    mlir ctx: ctx, blk: block do
      Func.call(args, callee: Attribute.flat_symbol_ref("#{name}", ctx: ctx)) >>>
        __MODULE__.dyn()
    end
  end

  defp gen_mlir(i, ctx, block) when is_integer(i) do
    mlir ctx: ctx, blk: block do
      __MODULE__.lit_int(value: Attribute.integer(Type.i64(), i)) >>> Type.i64()
    end
  end

  defp gen_mlir({:__block__, _, _} = ast, _, _) do
    ast
  end

  defp gen_mlir(ast, _, _) do
    ast = inspect(ast, pretty: true)
    Logger.info("AST ignored:\n#{ast}")
    ast
  end

  def from_ast(ast, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        Beaver.Slang.load(ctx, __MODULE__)

        mlir ctx: ctx do
          module do
            ast |> Macro.prewalk(&gen_mlir(&1, ctx, Beaver.Env.block()))
          end
        end
      end
    )
  end

  defmodule MaterializeBoundVariables do
    @moduledoc """
    erase `ex.var` and `ex.bind` ops, build usages in MLIR
    """
    use Beaver.MLIR.Pass, on: "func.func"

    defp extract_var_name(var_op) do
      Beaver.Walker.attributes(var_op)["name"]
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> MLIR.to_string()
    end

    defp lookup_var(op, acc) do
      [val] =
        Beaver.Walker.results(op)[0]
        |> Beaver.Walker.uses()
        |> Enum.to_list()

      case val |> MLIR.CAPI.mlirOpOperandGetOwner() |> MLIR.Operation.name() do
        "ex.bind" ->
          {op, acc}

        _ ->
          {Beaver.Walker.replace(op, acc.variables[extract_var_name(op)]), acc}
      end
    end

    def run(func, state) do
      func
      |> Beaver.Walker.postwalk(%{variables: %{}}, fn
        %MLIR.Operation{} = op, acc ->
          case MLIR.Operation.name(op) do
            "ex.bind" ->
              val = Beaver.Walker.operands(op)[1]
              1 = Beaver.Walker.uses(val) |> Enum.count()

              var =
                Beaver.Walker.operands(op)[0]
                |> MLIR.CAPI.mlirOpResultGetOwner()

              acc = put_in(acc, [:variables, extract_var_name(var)], val)
              r = Beaver.Walker.replace(op, val)
              MLIR.CAPI.mlirOperationDestroy(var)
              {r, acc}

            "ex.var" ->
              lookup_var(op, acc)

            _ ->
              {op, acc}
          end

        mlir, acc ->
          {mlir, acc}
      end)

      state
    end
  end
end
