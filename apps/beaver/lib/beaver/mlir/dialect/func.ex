defmodule Beaver.MLIR.Dialect.Func do
  use Beaver.MLIR.Dialect,
    dialect: "func",
    ops: Beaver.MLIR.Dialect.Registry.ops("func"),
    skips: ~w{func}

  defmodule Func do
    use Beaver.DSL.Op.Prototype, op_name: "func.func"
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

        arguments =
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

        Beaver.MLIR.Operation.create("func.func", arguments, Beaver.MLIR.__BLOCK__())
      end

    func_ast
  end
end
