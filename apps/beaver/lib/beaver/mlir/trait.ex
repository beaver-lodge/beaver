defmodule Beaver.MLIR.Trait do
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR

  @doc """
  Check if an op is a terminator. Note that this will return false if the op is unregistered when the MLIR context allows unregistered op.
  """
  def is_terminator?(op_name) do
    CAPI.beaverIsOpNameTerminator(MLIR.StringRef.create(op_name), MLIR.Managed.Context.get())
    |> Beaver.Native.to_term()
  end
end
