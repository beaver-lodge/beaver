defmodule Beaver.Env do
  @moduledoc """
  This module defines macros to getting MLIR context, region, block within the do block of mlir/1
  """

  @doc """
  return context in the DSL environment
  """
  defmacro context() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_ctx, nil}) do
      quote do
        Kernel.var!(beaver_internal_env_ctx)
      end
    else
      raise "no MLIR context in environment, maybe you forgot to put the ssa form inside the 'mlir ctx: ctx, do: ....' ?"
    end
  end

  defmacro region() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_env_region, nil}) do
      quote do
        Kernel.var!(beaver_env_region)
      end
    else
      quote do
        nil
      end
    end
  end

  defmacro block() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_block, nil}) do
      quote do
        Kernel.var!(beaver_internal_env_block)
      end
    else
      raise "no block in environment, maybe you forgot to put the ssa form inside the Beaver.mlir/2 macro or a block/1 macro?"
    end
  end

  defmacro block({var_name, _line, nil} = block_var) do
    if Macro.Env.has_var?(__CALLER__, {var_name, nil}) do
      quote do
        %Beaver.MLIR.Block{} = unquote(block_var)
      end
    else
      quote do
        Kernel.var!(unquote(block_var)) = Beaver.MLIR.Block.create([])
        %Beaver.MLIR.Block{} = Kernel.var!(unquote(block_var))
      end
    end
  end

  defmacro location() do
    raise "TODO: create location from Elixir caller"
  end
end
