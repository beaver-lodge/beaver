defmodule Beaver.Env do
  @moduledoc """
  Provide macros to insert MLIR context and IR element of structure. These macros are designed to mimic the behavior and aesthetics of __MODULE__/0, __CALLER__/0 in Elixir.
  Its distinguished form is to indicate this should not be expected to be a function or a macro works like a function.
  """

  defmacro mlir__CONTEXT__() do
    quote do
      Beaver.MLIR.Managed.Context.get()
    end
  end

  defmacro mlir__MODULE__() do
    quote do
      raise "TODO: impl me"
    end
  end

  defmacro mlir__PARENT__() do
    quote do
      raise "TODO: impl me"
    end
  end

  defmacro mlir__REGION__() do
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

  defmacro mlir__BLOCK__() do
    if Macro.Env.has_var?(__CALLER__, {:beaver_internal_env_block, nil}) do
      quote do
        Kernel.var!(beaver_internal_env_block)
      end
    else
      raise "no block in environment, maybe you forgot to put the ssa form inside the Beaver.mlir/2 macro or a block/1 macro?"
    end
  end

  defmacro mlir__BLOCK__({var_name, _line, nil} = block_var) do
    if Macro.Env.has_var?(__CALLER__, {var_name, nil}) do
      quote do
        %Beaver.MLIR.CAPI.MlirBlock{} = unquote(block_var)
      end
    else
      quote do
        Kernel.var!(unquote(block_var)) = Beaver.MLIR.Block.create([])
        %Beaver.MLIR.CAPI.MlirBlock{} = Kernel.var!(unquote(block_var))
      end
    end
  end

  defmacro mlir__LOCATION__() do
    Beaver.MLIR.Managed.Location.get()
  end
end
