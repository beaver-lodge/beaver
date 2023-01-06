defmodule Beaver.MLIR.Operation do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  import Beaver.MLIR.CAPI
  require Logger

  @doc false

  def create(%MLIR.Operation.State{} = state) do
    state |> MLIR.Operation.State.create() |> create
  end

  def create(state) do
    state |> Beaver.Native.ptr() |> Beaver.Native.bag(state) |> MLIR.CAPI.mlirOperationCreate()
  end

  defp create(op_name, %Beaver.DSL.SSA{
         block: %MLIR.CAPI.MlirBlock{} = block,
         arguments: arguments,
         results: results,
         filler: filler,
         ctx: ctx,
         loc: loc
       }) do
    filler =
      if is_function(filler, 0) do
        [regions: filler]
      else
        []
      end

    create_and_append(ctx, op_name, arguments ++ [result_types: results] ++ filler, block, loc)
  end

  # one single value, usually a terminator
  defp create(op_name, %MLIR.Value{} = op) do
    create(op_name, [op])
  end

  @doc false
  def create_and_append(
        %MLIR.CAPI.MlirContext{} = ctx,
        op_name,
        arguments,
        %MLIR.CAPI.MlirBlock{} = block,
        loc \\ nil
      )
      when is_list(arguments) do
    op = do_create(ctx, op_name, arguments, loc)
    Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(block, op)
    op
  end

  def results(%MLIR.CAPI.MlirOperation{} = op) do
    case CAPI.mlirOperationGetNumResults(op) |> Beaver.Native.to_term() do
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

  defp do_create(ctx, op_name, arguments, loc) when is_binary(op_name) and is_list(arguments) do
    location = loc || MLIR.Location.unknown()

    state = %MLIR.Operation.State{name: op_name, location: location, context: ctx}
    state = Enum.reduce(arguments, state, &MLIR.Operation.State.add_argument(&2, &1))

    state
    |> MLIR.Operation.State.create()
    |> create()
  end

  @default_verify_opts [dump: false, dump_if_fail: false]
  def verify!(op, opts \\ @default_verify_opts) do
    with {:ok, op} <-
           verify(op, opts ++ [should_raise: true]) do
      op
    else
      :null ->
        raise "MLIR operation verification failed because the operation is null. Maybe it is parsed from an ill-formed text format?"

      :fail ->
        raise "MLIR operation verification failed"
    end
  end

  def verify(op, opts \\ @default_verify_opts) do
    dump = opts |> Keyword.get(:dump, false)
    dump_if_fail = opts |> Keyword.get(:dump_if_fail, false)

    is_null = MLIR.is_null(op)

    if is_null do
      :null
    else
      is_success = from_module(op) |> MLIR.CAPI.mlirOperationVerify() |> Beaver.Native.to_term()

      if dump do
        Logger.warning("Start dumping op not verified. This might crash.")
        dump(op)
      end

      if is_success do
        {:ok, op}
      else
        if dump_if_fail do
          Logger.info("Start printing op failed to pass the verification. This might crash.")
          Logger.info(MLIR.to_string(op))
        end

        :fail
      end
    end
  end

  def dump(op) do
    op |> from_module |> mlirOperationDump()
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

  def from_module(module = %MLIR.Module{}) do
    CAPI.mlirModuleGetOperation(module)
  end

  def from_module(%CAPI.MlirOperation{} = op) do
    op
  end

  @doc false
  def eval_ssa(full_name, %Beaver.DSL.SSA{results: result_types} = ssa) do
    ssa =
      case result_types do
        [{:op, result_types}] ->
          %Beaver.DSL.SSA{ssa | results: List.wrap(result_types)}

        _ ->
          ssa
      end

    op = create(full_name, ssa)
    results = op |> results()

    case result_types do
      [{:op, _}] ->
        {op, results}

      _ ->
        results
    end
  end
end
