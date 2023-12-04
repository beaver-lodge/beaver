defmodule ElixirAST do
  @moduledoc "An example to showcase the Elixir AST dialect in IRDL test."
  use Beaver.Slang, name: "ex"
  alias Beaver.MLIR.Attribute

  defop defmodule(), do: []

  defp gen_mlir(
         {:defmodule, _,
          [
            {:__aliases__, [alias: false], [name]},
            [
              do: do_body
            ]
          ]} = ast
       ) do
    use Beaver
    ctx = MLIR.Context.create()
    Beaver.Slang.load(ctx, ElixirAST)

    mlir ctx: ctx do
      module do
        ElixirAST.defmodule name: "\"#{name}\"" do
          region do
            block functions do
            end
          end
        end >>>
          []
      end
      |> MLIR.dump!()
    end

    ast
  end

  defp gen_mlir(ast) do
    # ast |> dbg
    ast
  end

  def from_ast(ast) do
    ast
    |> Macro.prewalk(&gen_mlir/1)
  end
end
