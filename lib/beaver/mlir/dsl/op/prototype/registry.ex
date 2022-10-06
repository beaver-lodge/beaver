defmodule Beaver.MLIR.DSL.Op.Registry do
  use GenServer

  def start_link([]) do
    GenServer.start_link(__MODULE__, [])
  end

  def init(init_arg) do
    :ets.new(__MODULE__, [:public, :set, :named_table, read_concurrency: true])

    {:ok, init_arg}
  end

  def register(op_name, op_module) when is_binary(op_name) and is_atom(op_module) do
    :ets.insert(__MODULE__, {op_name, op_module})
  end

  def lookup(op_name) when is_binary(op_name) do
    case :ets.match(__MODULE__, {op_name, :"$1"}) do
      [[op_module]] ->
        op_module

      [] ->
        raise "unknown op: #{op_name}"
    end
  end
end
