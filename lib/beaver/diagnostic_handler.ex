defmodule Beaver.DiagnosticHandler do
  @moduledoc """
  Use process to run MLIR diagnostic error handler.

  The handler should be a function that takes a Beaver.MLIR.Diagnostic.t() and returns a value. The return value will be collected by the server and can be retrieved by calling `Beaver.DiagnosticHandler .collect/1`.
  Note that the diagnostic only lives within the lifecycle of the handler, keep it or send it to another process will lead to crash or other unexpected behavior.
  """
  alias Beaver.MLIR
  use GenServer

  defstruct fun: nil, return: nil, handler_state: nil

  defp noop(_d, acc), do: acc

  def init(fun) when is_function(fun, 2) do
    init({fn -> nil end, fun})
  end

  def init({init, fun}) when is_function(init, 0) do
    {:ok, %__MODULE__{handler_state: init.(), fun: fun || (&noop/2)}}
  end

  @doc """
  return the result of the handler
  """
  def collect(pid) when is_pid(pid) do
    GenServer.call(pid, :collect)
  end

  def handle_call(:collect, _from, %__MODULE__{return: return}) do
    {:reply, return, %__MODULE__{return: nil}}
  end

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
