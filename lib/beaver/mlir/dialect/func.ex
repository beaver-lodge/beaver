defmodule Beaver.MLIR.Dialect.Func do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  use Beaver.MLIR.Dialect,
    dialect: "func",
    ops: Beaver.MLIR.Dialect.Registry.ops("func")

  @doc """
  Syntax sugar for `Func.func` SSA expression. No need to specify result types.
  """
  defmacro func(call, opts) do
    quote do
      Beaver.MLIR.Dialect.Func.func_like(
        unquote(call),
        Beaver.MLIR.Dialect.Func.func(),
        unquote(opts)
      )
    end
  end

  @doc """
  Syntax sugar for `Func.func` alike SSA expression. No need to specify result types.
  """
  defmacro func_like(call, op, do: body) do
    {func_name, args} = call |> Macro.decompose_call()

    quote do
      unquote(args)
      |> List.wrap()
      |> List.flatten()
      |> Keyword.put_new(:sym_name, Beaver.MLIR.Attribute.string(unquote(func_name)))
      |> Keyword.put_new(:loc, Beaver.MLIR.Location.from_env(__ENV__))
      |> then(
        &Beaver.MLIR.Operation.create(%Beaver.SSA{
          op: unquote(op),
          ip: Beaver.Env.block(),
          ctx: Beaver.Env.context(),
          arguments: [fn -> unquote(body) end | &1]
        })
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

  def entry_block(op) do
    case Beaver.Walker.regions(op) |> Enum.to_list() do
      [r] -> r |> Beaver.Walker.blocks() |> Enum.at(0)
      _ -> nil
    end
  end
end
