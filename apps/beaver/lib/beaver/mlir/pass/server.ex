defmodule Beaver.MLIR.Pass.Server do
  alias Beaver.MLIR
  require MLIR.CAPI

  @moduledoc """
  server to forward invoking from C to a Elixir module
  """
  use GenServer

  @impl true
  def init(_) do
    {:ok, []}
  end

  @impl true
  def handle_info({:run, op_ref}, state) do
    %MLIR.CAPI.MlirOperation{ref: op_ref}
    |> MLIR.dump()

    {:noreply, state}
  end

  def handle_info(msg, state) do
    require Logger
    Logger.error("unhandled message in pass server: #{inspect(msg)}")

    {:noreply, state}
  end
end
