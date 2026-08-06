defmodule Beaver.MLIR.CompilationCache.Memory do
  @moduledoc "In-memory backend for incremental compilation artifacts."

  use GenServer

  @default_name __MODULE__

  def default_name, do: @default_name

  def start_link(opts \\ []) do
    {name, opts} = Keyword.pop(opts, :name)
    GenServer.start_link(__MODULE__, opts, if(name, do: [name: name], else: []))
  end

  def get(server \\ @default_name, key), do: call(server, {:get, key})
  def put(server \\ @default_name, key, value), do: call(server, {:put, key, value})
  def delete(server \\ @default_name, key), do: call(server, {:delete, key})
  def clear(server \\ @default_name), do: call(server, :clear)

  @impl GenServer
  def init(_opts), do: {:ok, %{}}

  @impl GenServer
  def handle_call({:get, key}, _from, state) do
    case Map.fetch(state, key) do
      {:ok, value} -> {:reply, {:ok, value}, state}
      :error -> {:reply, :miss, state}
    end
  end

  def handle_call({:put, key, value}, _from, state),
    do: {:reply, :ok, Map.put(state, key, value)}

  def handle_call({:delete, key}, _from, state),
    do: {:reply, :ok, Map.delete(state, key)}

  def handle_call(:clear, _from, _state), do: {:reply, :ok, %{}}

  defp call(server, message) do
    GenServer.call(server, message)
  catch
    :exit, reason -> {:error, reason}
  end
end
