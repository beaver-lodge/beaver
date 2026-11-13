alias Beaver.MLIR.Operation

defmodule Beaver.MLIR.ConditionallySpeculatable do
  @moduledoc """
  Installs and queries a callback-backed ConditionallySpeculatable fallback interface model.

  A dedicated attachment process owns the callback mailbox. Queries execute on the MLIR context worker pool, so no
  BEAM scheduler blocks while native MLIR waits for the callback result. The process is released with the context.
  """
  alias Beaver.MLIR
  @type speculatability :: :not_speculatable | :speculatable | :recursively_speculatable
  @callback speculatability(Operation.t()) :: speculatability()
  @spec attach(
          MLIR.Context.t(),
          String.t(),
          module() | (Operation.t() -> speculatability() | {:ok, speculatability()}),
          keyword()
        ) :: MLIR.ExternalInterface.Attachment.t()
  def attach(%MLIR.Context{} = ctx, op_name, implementation, opts \\ [])
      when is_binary(op_name) do
    callback = callback!(implementation)
    MLIR.ExternalInterface.attach(ctx, op_name, :conditionally_speculatable, %{get_speculatability: callback}, opts)
  end

  @doc false
  def callback!(implementation) when is_atom(implementation) do
    unless function_exported?(implementation, :speculatability, 1),
      do: raise(ArgumentError, "#{inspect(implementation)} must implement speculatability/1")

    &implementation.speculatability/1
  end

  def callback!(callback) when is_function(callback, 1), do: callback
  def callback!(other), do: raise(ArgumentError, "invalid ConditionallySpeculatable implementation: #{inspect(other)}")
  @spec query(Operation.t(), keyword()) :: speculatability()
  def query(%Operation{} = op, opts \\ []) do
    Beaver.Execution.run(MLIR.context(op), :speculatability, {op.ref},
      timeout: Keyword.get(opts, :timeout, 30_000) + 1_000
    )
    |> decode()
  end

  defp decode(0), do: :not_speculatable
  defp decode(1), do: :speculatable
  defp decode(2), do: :recursively_speculatable
end
