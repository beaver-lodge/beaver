defmodule Beaver.MLIR.Operation do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  import Beaver.MLIR.CAPI
  require Logger

  defp defer_if_terminator(arguments) do
    arguments
    |> Enum.reduce(
      true,
      fn
        # if it is real block, don't defer the creation
        {:successor, %Beaver.MLIR.CAPI.MlirBlock{}}, acc ->
          false && acc

        _, acc ->
          acc
      end
    )
  end

  @doc """
  Create a new operation from a operation state
  """
  def create(state) do
    state |> Exotic.Value.get_ptr() |> MLIR.CAPI.mlirOperationCreate()
  end

  @doc """
  Create a new operation from arguments and insert to managed insertion point
  """
  def create(op_name, %Beaver.DSL.SSA{arguments: arguments, results: results, filler: filler})
      when is_function(filler, 0) do
    create(op_name, arguments ++ [result_types: results, regions: filler])
  end

  def create(op_name, %Beaver.DSL.SSA{arguments: arguments, results: results}) do
    create(op_name, arguments ++ [result_types: results])
  end

  def create(op_name, %Beaver.DSL.Op.Prototype{
        operands: operands,
        attributes: attributes,
        results: results
      }) do
    create(op_name, operands ++ attributes ++ [result_types: results])
  end

  # one single value, usually a terminator
  def create(op_name, %MLIR.CAPI.MlirValue{} = op) do
    create(op_name, [op])
  end

  def create(op_name, arguments) do
    if MLIR.Trait.is_terminator?(op_name) && defer_if_terminator(arguments) do
      if block = MLIR.Managed.Block.get() do
        Beaver.MLIR.Managed.Terminator.defer(fn ->
          op = do_create(op_name, arguments)
          Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(block, op)
        end)

        {:deferred, {op_name, arguments}}
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

  def results(%MLIR.CAPI.MlirOperation{} = op) do
    case CAPI.mlirOperationGetNumResults(op) |> Exotic.Value.extract() do
      0 ->
        op

      1 ->
        CAPI.mlirOperationGetResult(op, 0)

      n when n > 1 ->
        for i <- 0..(n - 1)//1 do
          CAPI.mlirOperationGetResult(op, i)
        end
    end
  end

  def results({:deferred, {_func_name, _arguments}} = deferred) do
    deferred
  end

  defp do_create(op_name, arguments) when is_binary(op_name) and is_list(arguments) do
    location = MLIR.Managed.Location.get()

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
    |> MLIR.StringRef.Callback.collect_and_destroy()
  end

  @default_verify_opts [dump: false, dump_if_fail: false]
  def verify!(op, opts \\ @default_verify_opts) do
    with {:ok, op} <-
           verify(op, opts ++ [should_raise: true]) do
      op
    else
      :fail -> raise "MLIR operation verification failed"
    end
  end

  def verify(op, opts \\ @default_verify_opts) do
    dump = opts |> Keyword.get(:dump, false)
    dump_if_fail = opts |> Keyword.get(:dump_if_fail, false)
    is_success = MLIR.CAPI.mlirOperationVerify(op) |> Exotic.Value.extract()

    if dump do
      Logger.warning("Start dumping op not verified. This might crash.")
      dump(op)
    end

    if is_success do
      {:ok, op}
    else
      if dump_if_fail do
        Logger.warning("Start dumping op failed to pass the verification. This might crash.")
        dump(op)
      end

      :fail
    end
  end

  def dump(op) do
    mlirOperationDump(op)
    op
  end

  @doc """
  Verify the op and dump it. It raises if the verification fails.
  """
  def dump!(%MLIR.CAPI.MlirOperation{} = op) do
    verify!(op)
    mlirOperationDump(op)
    op
  end

  def name(%MLIR.CAPI.MlirOperation{} = operation) do
    MLIR.CAPI.mlirOperationGetName(operation)
    |> MLIR.CAPI.mlirIdentifierStr()
    |> MLIR.StringRef.extract()
  end

  def to_prototype(%MLIR.CAPI.MlirOperation{} = operation) do
    op_name = name(operation)

    struct!(MLIR.DSL.Op.Registry.lookup(op_name), %{
      operands: Beaver.Walker.operands(operation),
      attributes: Beaver.Walker.attributes(operation),
      results: Beaver.Walker.results(operation),
      successors: Beaver.Walker.successors(operation),
      regions: Beaver.Walker.regions(operation)
    })
  end

  def is_null(operation = %MLIR.CAPI.MlirOperation{}) do
    operation
    |> Exotic.Value.fetch(MLIR.CAPI.MlirOperation, :ptr)
    |> Exotic.Value.extract() == 0
  end

  def from_module(module = %CAPI.MlirModule{}) do
    CAPI.mlirModuleGetOperation(module)
  end
end
