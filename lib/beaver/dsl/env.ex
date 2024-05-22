defmodule Beaver.Env do
  @moduledoc """
  This module defines macros to getting MLIR context, region, block within the do block of mlir/1. It works like __MODULE__/0, __CALLER__/0 of Elixir special forms.
  """

  @doc """
  Return context in the DSL environment
  """
  defmacro context() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_ctx, nil}) do
      quote do
        match?(%Beaver.MLIR.Context{}, Kernel.var!(beaver_internal_env_ctx)) ||
          raise Beaver.EnvNotFoundError, Beaver.MLIR.Context

        Kernel.var!(beaver_internal_env_ctx)
      end
    else
      raise Beaver.EnvNotFoundError, Beaver.MLIR.Context
    end
  end

  @doc """
  Return region in the DSL environment
  """
  defmacro region() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_env_region, nil}) do
      quote do
        match?(%Beaver.MLIR.Region{}, Kernel.var!(beaver_env_region)) ||
          raise Beaver.EnvNotFoundError, Beaver.MLIR.Region

        Kernel.var!(beaver_env_region)
      end
    else
      # NOTE: will not raise error if region is not found, because macro mlir/2 doesn't work with region
      quote do
        Beaver.not_found(__ENV__)
      end
    end
  end

  @doc """
  Return block in the DSL environment
  """
  defmacro block() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_block, nil}) do
      quote do
        Kernel.var!(beaver_internal_env_block)
      end
    else
      raise Beaver.EnvNotFoundError, Beaver.MLIR.Block
    end
  end

  @doc """
  Return block with a given name in the DSL environment
  """
  defmacro block({var_name, _line, _} = block_var) do
    if Macro.Env.has_var?(__CALLER__, {var_name, nil}) do
      block_var
    else
      quote do
        Kernel.var!(unquote(block_var)) = Beaver.MLIR.Block.create()
      end
    end
  end
end
