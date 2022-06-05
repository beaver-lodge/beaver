defmodule Beaver.MLIR.Managed.Block do
  @moduledoc """
  Getting and pushing a managed block.
  """
  def get() do
    case Process.get(__MODULE__) do
      nil ->
        raise "no block set, create and call Beaver.MLIR.Managed.Block.push/1"

      [] ->
        raise "no block found, create and call Beaver.MLIR.Managed.Block.push/1"

      managed ->
        [head | _tail] = managed
        head
    end
  end

  def get(id) when is_atom(id) do
    Process.get({__MODULE__, id})
  end

  # TODO: add guard
  def push(block) do
    old =
      case Process.delete(__MODULE__) do
        nil ->
          []

        managed ->
          managed
      end

    new = [block | old]
    Process.put(__MODULE__, new)
    new
  end

  def push(id, block) when is_atom(id) do
    new = push(block)
    if {__MODULE__, id} in Process.get_keys(), do: raise("block already exists, #{id}")
    Process.put({__MODULE__, id}, block)
    new
  end

  def clear_ids() do
    if not empty?() do
      raise "some blocks still on the stack"
    end

    for key <- Process.get_keys() do
      with {__MODULE__, _id} <- key do
        Process.delete(key)
      end
    end
  end

  def empty?() do
    managed = Process.get(__MODULE__)
    managed == [] or is_nil(managed)
  end

  def pop() do
    managed = Process.get(__MODULE__)

    case managed do
      [] ->
        raise "no block found"

      nil ->
        raise "no block set"

      managed ->
        [head | tail] = managed
        Process.put(__MODULE__, tail)
        head
    end
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
