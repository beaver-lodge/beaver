defmodule Beaver.Env do
  alias Beaver.MLIR

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

  defmacro mlir__LOCATION__() do
    Beaver.MLIR.Managed.Location.get()
  end

  @doc """
  Setting a container. It could be a module/parent/region/block.
  When a higher-level container is set, all lower-level containers already set will be cleared.
  When a lower-level container is set, it will be checked it is under the higher-level containers.
  """
  def set_container(%MLIR.CAPI.MlirBlock{} = block) do
    MLIR.Managed.Block.set(block)
  end
end
