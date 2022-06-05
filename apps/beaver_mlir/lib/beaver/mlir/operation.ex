defmodule Beaver.MLIR.Operation do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI.IR
  alias Beaver.MLIR.CAPI
  import Beaver.MLIR.CAPI

  @doc """
  Create a new operation from a operation state
  """
  def create(state) do
    state |> Exotic.Value.get_ptr() |> IR.mlirOperationCreate()
  end

  @doc """
  Create a new operation from arguments and insert to managed insertion point
  """
  def create(op_name, arguments) do
    if MLIR.Trait.is_terminator?(op_name) do
      create_pending_terminator(op_name, arguments)
      # do_create(op_name, arguments)
    else
      do_create(op_name, arguments)
    end
  end

  defp create_pending_terminator(op_name, arguments) do
    # if MLIR.Trait.is_terminator?(op_name) do
    block = MLIR.Managed.Block.get()

    Beaver.MLIR.Managed.Terminator.defer(fn ->
      Beaver.MLIR.Managed.InsertionPoint.push(fn next_op ->
        Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(block, next_op)
      end)

      Beaver.MLIR.Managed.Block.push(block)
      do_create(op_name, arguments)
      Beaver.MLIR.Managed.Block.pop()
      Beaver.MLIR.Managed.InsertionPoint.pop()
    end)

    # Beaver.MLIR.Managed.InsertionPoint.push(fn _next_op ->
    #   raise "pending terminator found, can't insert after it"
    # end)
  end

  defp do_create(op_name, arguments) when is_binary(op_name) and is_list(arguments) do
    arguments =
      arguments
      |> Enum.filter(fn
        {key, _val} ->
          key != :block or key != :location

        _ ->
          true
      end)

    state = MLIR.Operation.State.get!(op_name, MLIR.Managed.Location.get())

    for argument <- arguments do
      MLIR.Operation.State.add_argument(state, argument)
    end

    op =
      state
      |> MLIR.Operation.create()

    ip = MLIR.Managed.InsertionPoint.pop()
    ip.(op)

    block = MLIR.Managed.Block.get()

    Beaver.MLIR.Managed.InsertionPoint.push(fn next_op ->
      Beaver.MLIR.CAPI.mlirBlockInsertOwnedOperationAfter(block, op, next_op)
    end)

    num_results = CAPI.mlirOperationGetNumResults(op) |> Exotic.Value.extract()

    if num_results == 1 do
      CAPI.mlirOperationGetResult(op, 0)
    else
      for i <- 0..(num_results - 1)//1 do
        CAPI.mlirOperationGetResult(op, i)
      end
    end
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

  def dump(op) do
    mlirOperationDump(op)
    op
  end

  def dump!(op) do
    verify!(op)
    mlirOperationDump(op)
    op
  end
end
