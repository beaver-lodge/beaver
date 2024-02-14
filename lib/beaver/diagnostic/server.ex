defmodule Beaver.Diagnostic.Server do
  @moduledoc """
  server to collect diagnostic printing from MLIR
  """
  use GenServer

  def init(_) do
    {:ok, ""}
  end

  def flush(pid) when is_pid(pid) do
    GenServer.call(pid, :flush)
  end

  def handle_call(:flush, _from, state) do
    reply = state
    new_state = ""
    {:reply, reply, new_state}
  end

  def handle_info(msg, state) do
    {:noreply, state <> msg}
  end
end
