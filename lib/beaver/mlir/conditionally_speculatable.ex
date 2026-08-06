defmodule Beaver.MLIR.ConditionallySpeculatable do
  @moduledoc """
  Installs and queries a callback-backed ConditionallySpeculatable fallback
  interface model.

  The attaching process owns the callback mailbox. Queries execute on the MLIR
  context worker pool, so no BEAM scheduler blocks while native MLIR waits for
  the callback result.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  alias Kinda.CallbackRuntime

  @type speculatability ::
          :not_speculatable | :speculatable | :recursively_speculatable

  @spec attach(
          MLIR.Context.t(),
          String.t(),
          (MLIR.Operation.t() -> speculatability() | {:ok, speculatability()}),
          keyword()
        ) :: term()
  def attach(%MLIR.Context{} = context, operation_name, callback, opts \\ [])
      when is_binary(operation_name) and is_function(callback, 1) do
    CAPI.beaver_raw_conditionally_speculatable_attach_fallback_model(
      context,
      operation_name,
      callback,
      Keyword.get(opts, :timeout, 30_000)
    )
  end

  @spec query(MLIR.Operation.t(), keyword()) :: speculatability()
  def query(%MLIR.Operation{ref: operation_ref}, opts \\ []) do
    timeout_ms = Keyword.get(opts, :timeout, 30_000)
    :ok = CAPI.beaver_raw_conditionally_speculatable_query_async(operation_ref)
    await_query(timeout_ms, nil)
  end

  defp await_query(timeout_ms, exception) do
    receive do
      {:get_speculatability, token, callback, _id, operation_term} ->
        outcome =
          CallbackRuntime.invoke_reply(
            token,
            fn -> invoke_callback(callback, Beaver.Native.check!(operation_term)) end,
            &reply/2
          )

        await_query(timeout_ms, exception || callback_exception(outcome))

      {:speculatability_done, value} ->
        finish_query(value, exception)
    after
      timeout_ms + 1_000 ->
        raise "timed out querying ConditionallySpeculatable fallback model"
    end
  end

  defp invoke_callback(callback, operation) do
    case callback.(operation) do
      value when value in [:not_speculatable, :speculatable, :recursively_speculatable] ->
        {:ok, value}

      {:ok, value}
      when value in [:not_speculatable, :speculatable, :recursively_speculatable] ->
        {:ok, value}

      other ->
        raise ArgumentError, "invalid speculatability callback result: #{inspect(other)}"
    end
  end

  defp reply(token, {:ok, value}) do
    CAPI.beaver_raw_callback_reply_code(token, true, encode(value))
  end

  defp reply(token, _outcome) do
    CAPI.beaver_raw_callback_reply_code(token, false, 0)
  end

  defp encode(:not_speculatable), do: 0
  defp encode(:speculatable), do: 1
  defp encode(:recursively_speculatable), do: 2

  defp decode(0), do: :not_speculatable
  defp decode(1), do: :speculatable
  defp decode(2), do: :recursively_speculatable

  defp callback_exception({:exception, _kind, _reason, _stacktrace} = exception),
    do: exception

  defp callback_exception(_outcome), do: nil

  defp finish_query(_value, {:exception, kind, reason, stacktrace}) do
    :erlang.raise(kind, reason, stacktrace)
  end

  defp finish_query(value, nil), do: decode(value)
end
