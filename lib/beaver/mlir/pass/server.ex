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
  def handle_info({:run, op_ref, pass_ref, token_ref}, %{run: run} = state) do
    op = %MLIR.Operation{ref: op_ref}
    pass = %MLIR.CAPI.MlirExternalPass{ref: pass_ref}

    try do
      case run.(op) do
        :ok ->
          :ok

        :error ->
          MLIR.CAPI.mlirExternalPassSignalFailure(pass)

        other ->
          MLIR.CAPI.mlirExternalPassSignalFailure(pass)

          Logger.error(
            "must return :ok or :error in run/1 of the pass #{__MODULE__}, get: #{inspect(other)}"
          )
      end
    rescue
      exception ->
        MLIR.CAPI.mlirExternalPassSignalFailure(pass)

        Logger.error("fail to run a pass.")

        Logger.error(Exception.format(:error, exception, __STACKTRACE__))
    after
      :ok = MLIR.CAPI.beaver_raw_pass_token_signal(token_ref)
    end

    {:noreply, state}
  end

  def handle_info(msg, state) do
    Logger.error("unhandled message in pass server: #{inspect(msg)}")

    {:noreply, state}
  end
end
