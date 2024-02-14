defmodule Beaver.Diagnostic.Server do
  @moduledoc """
  server to collect diagnostic printing from MLIR
  """
  use GenServer
  require Logger

  def init(_) do
    {:ok, ""}
  end

  def flush(%Beaver.MLIR.Context{__diagnostic_server__: pid}) when is_pid(pid) do
    flush(pid)
  end

  def flush(pid) when is_pid(pid) do
    GenServer.call(pid, :flush)
  end

  def flush(_) do
    raise "To flush diagnostics, you need to create the MLIR context with a diagnostic server. To use the built-in one: 'MLIR.Context.create(diagnostic: :server)'"
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
