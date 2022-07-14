defmodule Beaver.MLIR.Managed.Block do
  @moduledoc """
  Getting and pushing a managed block.
  """

  @doc false
  def set(block) do
    Process.put(__MODULE__, block)
  end

  def clear() do
    Process.delete(__MODULE__)
  end
end
