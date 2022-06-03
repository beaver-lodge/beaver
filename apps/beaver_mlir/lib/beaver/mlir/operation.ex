defmodule Beaver.MLIR.Operation do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI.IR
  import Beaver.MLIR.CAPI

  @doc """
  Create a new operation from a operation state
  """
  def create(state) do
    state |> Exotic.Value.get_ptr() |> IR.mlirOperationCreate()
  end

  @doc """
  Print operation to a Elixir `String`. This function could be very expensive. It is recommended to use it at compile-time or debugging.
  """
  def to_string(operation) do
    string_ref_callback_closure = MLIR.StringRef.Callback.create()

    MLIR.CAPI.mlirOperationPrint(
      operation,
      Exotic.Value.as_ptr(string_ref_callback_closure),
      Exotic.Value.Ptr.null()
    )

    string_ref_callback_closure
    |> MLIR.StringRef.Callback.collect_and_destory()
  end

  def verify!(op) do
    is_success = IR.mlirOperationVerify(op) |> Exotic.Value.extract()

    if not is_success do
      raise "MLIR operation verification failed"
    end
  end

  def dump!(op) do
    verify!(op)
    mlirOperationDump(op)
  end
end
