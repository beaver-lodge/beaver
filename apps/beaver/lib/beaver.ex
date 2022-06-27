defmodule Beaver do
  @moduledoc """
  This module contains top level functions and macros for Beaver DSL for MLIR.
  """

  @doc """
  This is a macro where Beaver's MLIR DSL expressions get transformed to MLIR API calls
  """
  defmacro mlir(do: block) do
    new_block_ast = Beaver.DSL.transform_ssa(block)

    quote do
      unquote(new_block_ast)
    end
  end

  defmacro mlir_debug(do: block) do
    new_block_ast = Beaver.DSL.transform_ssa(block)

    env = __CALLER__
    new_block_ast |> Macro.to_string() |> IO.puts()

    block
    |> Macro.expand(env)
    # |> Macro.prewalk(&Macro.expand(&1, env))
    |> Macro.to_string()
    |> IO.puts()

    quote do
      unquote(new_block_ast)
    end
  end
end
