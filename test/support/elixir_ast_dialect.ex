defmodule ElixirAST do
  require Logger
  @moduledoc "Prototype the Elixir AST dialect."
  use Beaver.Slang, name: "ex"
  deftype dyn
  deftype bound
  deftype unbound
  defop mod(), do: []
  defop func(), do: []
  defop lit_int(), do: [Type.i64()]
  defop var(), do: []
  defop bind(), do: []
  defop call(), do: []
  defop add(), do: []
  alias Beaver.MLIR.Type
  alias Beaver.MLIR.Attribute
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
      __MODULE__.mod name: "\"#{name}\"" do
        region do
          block functions do
            Macro.prewalk(do_body, &gen_mlir(&1, ctx, Beaver.Env.block()))
          end
        end
      end >>>
        []
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
      __MODULE__.func name: "\"#{name}\"" do
        region do
          block func_body do
            Macro.prewalk(do_body, &gen_mlir(&1, ctx, Beaver.Env.block()))
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
      __MODULE__.call(args, name: "\"#{name}\"") >>> __MODULE__.dyn()
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
            ast
            |> Macro.prewalk(&gen_mlir(&1, ctx, Beaver.Env.block()))
          end
        end
      end
    )
  end
end
