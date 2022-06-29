defmodule Beaver.MLIR.Operation do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI.IR
  alias Beaver.MLIR.CAPI
  import Beaver.MLIR.CAPI
  require Logger

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
    defer_if_terminator = Keyword.get(arguments, :defer_if_terminator, true)

    if defer_if_terminator and MLIR.Trait.is_terminator?(op_name) do
      if block = MLIR.Managed.Block.get() do
        Beaver.MLIR.Managed.Terminator.defer(fn ->
          op = do_create(op_name, arguments)
          Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(block, op)
        end)

        :deferred
      else
        raise "deferred terminator creation requires a block"
      end
    else
      op = do_create(op_name, arguments)

      if block = MLIR.Managed.Block.get() do
        Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(block, op)
      end

      op
    end
  end

  # TODO: add guard
  def results(op) do
    num_results = CAPI.mlirOperationGetNumResults(op) |> Exotic.Value.extract()

    if num_results == 1 do
      CAPI.mlirOperationGetResult(op, 0)
    else
      for i <- 0..(num_results - 1)//1 do
        CAPI.mlirOperationGetResult(op, i)
      end
    end
  end

  defp do_create(op_name, arguments) when is_binary(op_name) and is_list(arguments) do
    {_ctx, arguments} = arguments |> Keyword.pop(:mlir_ctx)
    {block, arguments} = arguments |> Keyword.pop(:mlir_block)
    {location, arguments} = arguments |> Keyword.pop(:mlir_location)

    block = block || MLIR.Managed.Block.get()
    location = location || MLIR.Managed.Location.get()

    state = MLIR.Operation.State.get!(op_name, location)

    for argument <- arguments do
      MLIR.Operation.State.add_argument(state, argument)
    end

    state
    |> MLIR.Operation.create()
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

  def verify!(op, opts \\ [dump: false, dump_if_fail: false]) do
    dump = opts |> Keyword.get(:dump, false)
    dump_if_fail = opts |> Keyword.get(:dump_if_fail, false)
    is_success = IR.mlirOperationVerify(op) |> Exotic.Value.extract()

    if dump do
      Logger.error("Start dumping op not verified. This might crash.")
      dump(op)
    end

    if not is_success do
      if dump_if_fail do
        Logger.error("Start dumping op fails in the verification. This might crash.")
        dump(op)
      end

      raise "MLIR operation verification failed"
    end

    op
  end

  def dump(op) do
    mlirOperationDump(op)
    op
  end

  @doc """
  Verify the op and dump it. It raises if the verification fails.
  """
  def dump!(op) do
    verify!(op)
    mlirOperationDump(op)
    op
  end
end
