defmodule Beaver.MLIR.Managed.Block do
  @moduledoc """
  Getting and pushing a managed block.
  """
  def get() do
    Process.get(__MODULE__)
  end

  @doc false
  def set(block) do
    Process.put(__MODULE__, block)
  end

  def clear() do
    Process.delete(__MODULE__)
  end

  def from_opts(opts) when is_list(opts) do
    block = opts[:block]

    if block do
      block
    else
      get()
    end
  end
end
