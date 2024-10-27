defmodule Beaver.PassRunner do
  alias Beaver.MLIR

  require Logger

  @moduledoc """
  `GenServer` to run an MLIR pass implemented in Elixir
  """
  use GenServer

  @impl true
  def init(fun) do
    {:ok, %{run: fun}}
  end

  @impl true
  def handle_info({:run, op_ref, token_ref}, %{run: run} = state) do
    try do
      run.(%MLIR.Operation{ref: op_ref})
      MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref)
    rescue
      exception ->
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_failure(token_ref)
        reraise exception, __STACKTRACE__
    end

    {:noreply, state}
  end
end
