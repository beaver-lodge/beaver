defmodule Beaver.DiagnosticHandlerRunner do
  alias Beaver.MLIR
  use GenServer

  @moduledoc """
  `GenServer` to run MLIR diagnostic error handler.

  The handler should be a function that takes a `MLIR.Diagnostic.t()` and returns a value. The return value will be collected by the server and can be retrieved by calling `Beaver.DiagnosticHandlerRunner.collect/1`.

  > #### Diagnostic's lifecycle {: .warning}
  >
  > Note that the diagnostic only lives within the lifecycle of the handler. Prolonging its lifetime by keeping it or sending it to another process will lead to BEAM crash or other unexpected behavior.
  """
  defstruct fun: nil, return: nil, handler_state: nil

  @doc """
  return the result of the handler
  """
  def collect(pid) when is_pid(pid) do
    GenServer.call(pid, :collect)
  end

  @doc """
  Attach the diagnostic handler runner to the context and return the handler id.
  """
  def attach(%MLIR.Context{ref: ctx_ref}, handler) when is_pid(handler) do
    MLIR.CAPI.beaver_raw_context_attach_diagnostic_handler(ctx_ref, handler)
  end

  @impl true
  def init(fun) when is_function(fun, 2) do
    init({fn -> nil end, fun})
  end

  def init({init, fun}) when is_function(init, 0) do
    {:ok, %__MODULE__{handler_state: init.(), fun: fun}}
  end

  @impl true
  def handle_call(:collect, _from, %__MODULE__{return: return}) do
    {:reply, return, %__MODULE__{return: nil}}
  end

  @impl true
  def handle_info(
        {:diagnostic, ref, token_ref},
        %__MODULE__{fun: fun, handler_state: handler_state} = state
      ) do
    try do
      fun.(%MLIR.Diagnostic{ref: ref}, handler_state)
      |> tap(fn _ ->
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_success(token_ref)
      end)
    rescue
      e ->
        MLIR.CAPI.beaver_raw_logical_mutex_token_signal_failure(token_ref)
        reraise e, __STACKTRACE__
    end
    |> then(&{:noreply, %__MODULE__{state | return: &1}})
  end
end
