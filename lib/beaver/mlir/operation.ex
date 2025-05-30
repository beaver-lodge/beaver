defmodule Beaver.MLIR.Operation do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias __MODULE__.State
  alias Beaver.Changeset
  import Beaver.MLIR.CAPI
  require Logger
  @behaviour Access

  use Kinda.ResourceKind, forward_module: Beaver.Native

  def create(%Beaver.SSA{
        op: op_name,
        blk: %MLIR.Block{} = block,
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
    state |> Beaver.Native.ptr() |> mlirOperationCreate()
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

  def name(%__MODULE__{} = operation) do
    mlirOperationGetName(operation)
    |> mlirIdentifierStr()
    |> MLIR.to_string()
  end

  defdelegate location(op), to: MLIR.CAPI, as: :mlirOperationGetLocation
  defdelegate parent(op), to: MLIR.CAPI, as: :mlirOperationGetParentOperation

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

  @impl Access
  def fetch(operation, attribute) do
    attr = mlirOperationGetAttributeByName(operation, MLIR.StringRef.create(attribute))

    if MLIR.null?(attr) do
      :error
    else
      {:ok, attr}
    end
  end

  @impl Access
  def get_and_update(operation, attribute, function) do
    attr =
      case fetch(operation, attribute) do
        {:ok, attr} -> attr
        :error -> nil
      end

    case function.(attr) do
      {_current_value, new_value} ->
        ctx = MLIR.context(operation)

        mlirOperationSetAttributeByName(
          operation,
          MLIR.StringRef.create(attribute),
          Beaver.Deferred.create(new_value, ctx)
        )

      :pop ->
        mlirOperationRemoveAttributeByName(operation, MLIR.StringRef.create(attribute))
    end

    {attr, operation}
  end

  @impl Access
  def pop(operation, attribute) do
    {:ok, attr} = fetch(operation, attribute)
    mlirOperationRemoveAttributeByName(operation, MLIR.StringRef.create(attribute))
    {attr, operation}
  end

  def with_symbol_table(%__MODULE__{} = op, fun) do
    symbol_table = mlirSymbolTableCreate(op)

    try do
      fun.(symbol_table)
    after
      mlirSymbolTableDestroy(symbol_table)
    end
  end
end
