defmodule Beaver.MLIR.Dialect.Func do
  use Beaver.MLIR.Dialect,
    dialect: "func",
    ops: Beaver.MLIR.Dialect.Registry.ops("func") |> Enum.reject(fn x -> x in ~w{func} end)

  def op_module_names() do
    [__MODULE__.Func]
  end

  defmacro func(call, do: block) do
    {func_name, args} = call |> Macro.decompose_call()
    if not is_atom(func_name), do: raise("func name must be an atom")

    func_ast =
      quote do
        # create function

        if not is_list(unquote_splicing(args)),
          do: raise("argument of Func.func must be a keyword")

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

        Beaver.MLIR.Operation.create_and_append(
          Beaver.MLIR.__CONTEXT__(),
          "func.func",
          arguments,
          Beaver.MLIR.__BLOCK__()
        )
      end

    func_ast
  end
end

require Beaver.MLIR.Dialect

Beaver.MLIR.Dialect.define_op_modules(
  Beaver.MLIR.Dialect.Func,
  "func",
  Beaver.MLIR.Dialect.Registry.ops("func")
)
