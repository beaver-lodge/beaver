defmodule Beaver.MLIR.Dialect.Func do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect

  use Dialect.Generator,
    dialect: "func",
    ops: Dialect.Registry.ops("func") |> Enum.reject(fn x -> x in ~w{func} end)

  defmodule Func do
    def create(arguments) do
      MLIR.Operation.create("func.func", arguments)
    end

    @behaviour Beaver.DSL.Op.Prototype

    @impl true
    def op_name() do
      "func.func"
    end
  end

  defmacro func(call, do: block) do
    {func_name, args} = call |> Macro.decompose_call()
    if not is_atom(func_name), do: raise("func name must be an atom")

    func_ast =
      quote do
        # TODO: support getting ctx from opts
        ctx = Beaver.MLIR.Managed.Context.get()

        # create function

        if not is_list(unquote_splicing(args)),
          do: raise("augument of Func.func must be a keyword")

        Beaver.MLIR.Dialect.Func.Func.create(
          Enum.uniq_by(
            unquote_splicing(args) ++
              [
                sym_name: "\"#{unquote(func_name)}\"",
                regions: fn ->
                  unquote(block)
                end
              ],
            fn {x, _} -> x end
          )
        )
      end

    func_ast
  end
end
