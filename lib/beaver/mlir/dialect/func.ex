defmodule Beaver.MLIR.Dialect.Func do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  use Beaver.MLIR.Dialect,
    dialect: "func",
    ops: Beaver.MLIR.Dialect.Registry.ops("func")

  defmacro func(call, do: body) do
    {func_name, args} = call |> Macro.decompose_call()

    quote do
      unquote(args)
      |> List.wrap()
      |> List.flatten()
      |> Keyword.put_new(:sym_name, Beaver.MLIR.Attribute.string(unquote(func_name)))
      |> Keyword.put_new(:loc, Beaver.MLIR.Location.from_env(__ENV__))
      |> then(
        &Beaver.MLIR.Operation.create_and_append(
          Beaver.Env.context(),
          "func.func",
          [fn -> unquote(body) end | &1],
          [],
          Beaver.Env.block()
        )
      )
    end
  end

  def external?(op) do
    with [r] <- Beaver.Walker.regions(op) |> Enum.to_list(),
         [] <- Beaver.Walker.blocks(r) |> Enum.to_list() do
      true
    else
      _ -> false
    end
  end
end
