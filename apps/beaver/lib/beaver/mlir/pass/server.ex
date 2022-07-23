defmodule Beaver.MLIR.Pass.Server do
  alias Beaver.MLIR
  require MLIR.CAPI
  require Logger

  @moduledoc """
  server to forward invoking from C to a Elixir module
  """
  use GenServer

  @impl true
  def init(_) do
    {:ok, []}
  end

  @impl true
  def handle_info({:run, op_ref, token_ref}, state) do
    str =
      %MLIR.CAPI.MlirOperation{ref: op_ref}
      |> MLIR.to_string()

    Logger.debug("op in pass: #{str}")

    :ok = MLIR.CAPI.beaver_raw_pass_token_signal(token_ref)
    Logger.debug("beaver_raw_pass_token_signal: #{:ok}")
    {:noreply, state}
  end

  def handle_info(msg, state) do
    Logger.error("unhandled message in pass server: #{inspect(msg)}")

    {:noreply, state}
  end
end
