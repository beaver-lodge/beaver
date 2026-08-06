defmodule Beaver.MLIR.ConditionallySpeculatable do
  @moduledoc """
  Installs and queries a callback-backed ConditionallySpeculatable fallback
  interface model.

  A dedicated attachment process owns the callback mailbox. Queries execute on
  the MLIR context worker pool, so no BEAM scheduler blocks while native MLIR
  waits for the callback result. The process is released with the context.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @type speculatability ::
          :not_speculatable | :speculatable | :recursively_speculatable

  @callback speculatability(MLIR.Operation.t()) :: speculatability()

  @spec attach(
          MLIR.Context.t(),
          String.t(),
          module() | (MLIR.Operation.t() -> speculatability() | {:ok, speculatability()}),
          keyword()
        ) :: MLIR.ExternalInterface.Attachment.t()
  def attach(%MLIR.Context{} = context, operation_name, implementation, opts \\ [])
      when is_binary(operation_name) do
    callback = callback!(implementation)

    MLIR.ExternalInterface.attach(
      context,
      operation_name,
      :conditionally_speculatable,
      %{get_speculatability: callback},
      opts
    )
  end

  @doc false
  def callback!(implementation) when is_atom(implementation) do
    unless function_exported?(implementation, :speculatability, 1) do
      raise ArgumentError,
            "#{inspect(implementation)} must implement speculatability/1"
    end

    &implementation.speculatability/1
  end

  def callback!(callback) when is_function(callback, 1), do: callback

  def callback!(other),
    do:
      raise(ArgumentError, "invalid ConditionallySpeculatable implementation: #{inspect(other)}")

  @spec query(MLIR.Operation.t(), keyword()) :: speculatability()
  def query(%MLIR.Operation{ref: operation_ref}, opts \\ []) do
    timeout_ms = Keyword.get(opts, :timeout, 30_000)
    :ok = CAPI.beaver_raw_conditionally_speculatable_query_async(operation_ref)
    await_query(timeout_ms)
  end

  defp await_query(timeout_ms) do
    receive do
      {:speculatability_done, value} ->
        decode(value)
    after
      timeout_ms + 1_000 ->
        raise "timed out querying ConditionallySpeculatable fallback model"
    end
  end

  defp decode(0), do: :not_speculatable
  defp decode(1), do: :speculatable
  defp decode(2), do: :recursively_speculatable
end
