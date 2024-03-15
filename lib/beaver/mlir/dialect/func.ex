defmodule Beaver.MLIR.Dialect.Func do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  use Beaver.MLIR.Dialect,
    dialect: "func",
    ops: Beaver.MLIR.Dialect.Registry.ops("func")

  defmacro func(call, do: body) do
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
                  unquote(body)
                end
              ],
            fn {x, _} -> x end
          )

        location =
          Keyword.get(arguments, :loc) ||
            Beaver.MLIR.Location.file(
              name: __ENV__.file,
              line: __ENV__.line,
              ctx: Beaver.Env.context()
            )

        Beaver.MLIR.Operation.create_and_append(
          Beaver.Env.context(),
          "func.func",
          arguments,
          Beaver.Env.block()
        )
      end

    func_ast
  end
end
