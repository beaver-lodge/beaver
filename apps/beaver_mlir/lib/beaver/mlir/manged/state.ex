defmodule Beaver.MLIR.Managed.Terminator do
  @moduledoc """
  Pending terminator creators
  """

  def defer(creator) when is_function(creator) do
    old = Process.delete(__MODULE__) || []
    new = old ++ [creator]
    Process.put(__MODULE__, new)
    new
  end

  def resolve() do
    all = Process.delete(__MODULE__) || []

    with key = {__MODULE__.Block, _} <- Process.get_keys() do
      Process.delete(key)
    end

    Process.delete({__MODULE__.Blocks}) || []

    for f <- all do
      f.()
    end
  end

  def put_block(id, block) do
    Process.put({__MODULE__.Block, id}, block)
  end

  def get_block(id) do
    Process.get({__MODULE__.Block, id})
  end
end
