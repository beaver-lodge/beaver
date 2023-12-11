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
    mlir ctx: ctx, block: block do
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
    mlir ctx: ctx, block: block do
      Func.func sym_name: "\"#{name}\"", function_type: Type.function([], [dyn()]) do
        region do
          block func_body do
            %MLIR.Value{} =
              ret_value =
              case Macro.prewalk(do_body, &gen_mlir(&1, ctx, Beaver.Env.block())) do
                {:__block__, [], values} ->
                  values |> List.last()

                %Beaver.MLIR.Value{} = v ->
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
    mlir ctx: ctx, block: block do
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
    mlir ctx: ctx, block: block do
      [left, right] =
        for i <- [left, right] do
          Macro.prewalk(i, &gen_mlir(&1, ctx, block))
        end

      __MODULE__.add(left, right) >>> __MODULE__.dyn()
    end
  end

  defp gen_mlir({name, [], mod}, ctx, block) when is_atom(name) and is_atom(mod) do
    mlir ctx: ctx, block: block do
      __MODULE__.var(name: "\"#{name}\"") >>> __MODULE__.dyn()
    end
  end

  defp gen_mlir({name, [], [] = args}, ctx, block) when is_atom(name) do
    mlir ctx: ctx, block: block do
      Func.call(args, callee: MLIR.Attribute.flat_symbol_ref("#{name}", ctx: ctx)) >>>
        __MODULE__.dyn()
    end
  end

  defp gen_mlir(i, ctx, block) when is_integer(i) do
    mlir ctx: ctx, block: block do
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

  defmodule UseBoundVariables do
    @moduledoc false
    use Beaver.MLIR.Pass, on: "func.func"

    def run(op) do
      op
      |> Beaver.Walker.postwalk(%{}, fn
        op = %MLIR.Operation{}, acc ->
          case MLIR.Operation.name(op) do
            "ex.bind" ->
              MLIR.dump(op)

              var = Beaver.Walker.operands(op)[1]
              Beaver.Walker.uses(var) |> Enum.to_list() |> Enum.map(&dbg/1)
              {Beaver.Walker.replace(op, var), acc}

            "ex.var" ->
              MLIR.dump(op)
              {op, acc}

            _ ->
              {op, acc}
          end

          {op, acc}

        mlir, acc ->
          {mlir, acc}
      end)

      :ok
    end
  end
end
