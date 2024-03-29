defmodule Beaver.MLIR.Operation do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias __MODULE__.{State, Changeset}
  import Beaver.MLIR.CAPI
  require Logger

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  def create(%Beaver.SSA{
        op: op_name,
        block: %MLIR.Block{} = block,
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

  def create(%Changeset{} = c) do
    c |> State.create() |> create
  end

  def create(%State{} = state) do
    state |> Beaver.Native.ptr() |> Beaver.Native.bag(state) |> mlirOperationCreate()
  end

  @doc false
  def create_and_append(
        %MLIR.Context{} = ctx,
        op_name,
        arguments,
        %MLIR.Block{} = block,
        loc \\ nil
      )
      when is_list(arguments) do
    op = do_create(ctx, op_name, arguments, loc)
    mlirBlockAppendOwnedOperation(block, op)
    op
  end

  def results(%__MODULE__{} = op) do
    case mlirOperationGetNumResults(op) |> Beaver.Native.to_term() do
      0 ->
        op

      1 ->
        mlirOperationGetResult(op, 0)

      n when n > 1 ->
        for i <- 0..(n - 1)//1 do
          mlirOperationGetResult(op, i)
        end
    end
  end

  def results({:deferred, {_func_name, _arguments}} = deferred) do
    deferred
  end

  defp do_create(ctx, op_name, arguments, loc) when is_binary(op_name) and is_list(arguments) do
    location = loc || MLIR.Location.unknown()
    changeset = %Changeset{name: op_name, location: location, context: ctx}

    Enum.reduce(arguments, changeset, &Changeset.add_argument(&2, &1))
    |> State.create()
    |> create()
  end

  @default_verify_opts [debug: false]
  def verify!(op, opts \\ @default_verify_opts) do
    case verify(op, opts ++ [should_raise: true]) do
      {:ok, op} ->
        op

      :null ->
        raise "MLIR operation verification failed because the operation is null. Maybe it is parsed from an ill-formed text format? Please have a look at the diagnostic output above by MLIR C++"

      :fail ->
        raise "MLIR operation verification failed"
    end
  end

  def verify(op, opts \\ @default_verify_opts) do
    debug = opts |> Keyword.get(:debug, false)

    is_null = MLIR.is_null(op)

    if is_null do
      :null
    else
      is_success = from_module(op) |> mlirOperationVerify() |> Beaver.Native.to_term()

      if not is_success and debug do
        Logger.info("Start printing op failed to pass the verification. This might crash.")
        Logger.info(MLIR.to_string(op))
      end

      if is_success do
        {:ok, op}
      else
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
  def dump!(%__MODULE__{} = op) do
    verify!(op)
    mlirOperationDump(op)
    op
  end

  def name(%__MODULE__{} = operation) do
    mlirOperationGetName(operation)
    |> mlirIdentifierStr()
    |> MLIR.StringRef.to_string()
  end

  def from_module(%MLIR.Module{} = module) do
    mlirModuleGetOperation(module)
  end

  def from_module(%__MODULE__{} = op) do
    op
  end

  @doc false
  def eval_ssa(%Beaver.SSA{results: result_types} = ssa) do
    ssa =
      case result_types do
        [{:op, result_types}] ->
          %Beaver.SSA{ssa | results: List.wrap(result_types)}

        _ ->
          ssa
      end

    op = create(ssa)
    results = op |> results()

    case result_types do
      [{:op, _}] ->
        {op, results}

      _ ->
        results
    end
  end
end
