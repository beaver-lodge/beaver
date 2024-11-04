defmodule Beaver.PassRunner do
  alias Beaver.MLIR

  @moduledoc """
  `GenServer` to run an MLIR pass implemented in Elixir
  """
  use GenServer

  def start_link([run | opts]) do
    GenServer.start_link(__MODULE__, run, opts)
  end

  @impl true
  def init(run) do
    {:ok, %{run: run}}
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
