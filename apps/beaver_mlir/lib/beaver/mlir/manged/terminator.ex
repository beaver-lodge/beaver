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

    for f <- all do
      f.()
    end

    for key <- Process.get_keys() do
      with {__MODULE__.Block, _} <- key do
        Process.delete(key)
      end
    end
  end

  def put_block(id, block) do
    Process.put({__MODULE__.Block, id}, block)
  end

  def get_block(id) do
    block = Process.get({__MODULE__.Block, id})

    if is_nil(block) do
      raise "Block #{id} not found"
    end

    block
  end
end
