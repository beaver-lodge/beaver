defmodule Beaver.Diagnostic.Server do
  @moduledoc """
  server to collect diagnostic printing from MLIR
  """
  use GenServer
  require Logger

  def init(init_arg) do
    {:ok, init_arg}
  end

  def handle_info(msg, state) do
    IO.write(msg)
    Logger.error(msg)
    {:noreply, state}
  end
end
