defmodule Beaver.MLIR.Pass.Server do
  alias Beaver.MLIR
  require MLIR.CAPI
  require Logger

  @moduledoc """
  server to forward invoking from C to a Elixir module
  """
  use GenServer

  @impl true
  def init(opts) do
    {:ok, Map.new(opts)}
  end

  @impl true
  def handle_info({:run, op_ref, pass_ref, token_ref}, %{pass_module: pass_module} = state) do
    op = %MLIR.CAPI.MlirOperation{ref: op_ref}
    pass = %MLIR.CAPI.MlirExternalPass{ref: pass_ref}

    with :ok <- apply(pass_module, :run, [op]) do
      :ok
    else
      :error ->
        MLIR.CAPI.mlirExternalPassSignalFailure(pass)

      other ->
        raise "must return :ok or :error in run/1 of the pass #{__MODULE__}, get: #{inspect(other)}"
    end

    :ok = MLIR.CAPI.beaver_raw_pass_token_signal(token_ref)
    {:noreply, state}
  end

  def handle_info(msg, state) do
    Logger.error("unhandled message in pass server: #{inspect(msg)}")

    {:noreply, state}
  end
end
