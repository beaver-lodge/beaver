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
        blk: block,
        arguments: arguments,
        results: results,
        filler: filler,
        ctx: ctx,
        loc: loc
      }) do
    filler =
      if is_function(filler, 0) do
        List.wrap(filler)
      else
        []
      end

    location = loc || MLIR.Location.unknown()
    changeset = %Changeset{name: op_name, location: location, context: ctx}

    Enum.reduce(arguments ++ filler, changeset, &Changeset.add_argument(&2, &1))
    |> then(fn changeset -> Enum.reduce(results, changeset, &Changeset.add_result(&2, &1)) end)
    |> State.create()
    |> create()
    |> tap(
      &case block do
        %MLIR.Block{} -> MLIR.Block.append(block, &1)
        {:not_found, _} -> nil
        nil -> nil
      end
    )
  end

  def create(%Changeset{} = c) do
    c |> State.create() |> create
  end

  def create(%State{} = state) do
    state |> Beaver.Native.ptr() |> mlirOperationCreate()
  end

  @doc """
  Normalize the results of the given operation in the following way:
  - If the operation has no result, return the operation itself.
  - If the operation has one result, return that result.
  - If the operation has multiple results, return a list of results.
  """
  def results(%__MODULE__{} = op) do
    case mlirOperationGetNumResults(op) |> Beaver.Native.to_term() do
      0 ->
        op

      1 ->
        result(op, 0)

      n when n > 1 ->
        for i <- 0..(n - 1)//1 do
          result(op, i)
        end
    end
  end

  def results({:deferred, {_func_name, _arguments}} = deferred) do
    deferred
  end

  def name(%__MODULE__{} = operation) do
    mlirOperationGetName(operation)
    |> mlirIdentifierStr()
    |> MLIR.to_string()
  end

  defdelegate location(op), to: MLIR.CAPI, as: :mlirOperationGetLocation
  defdelegate parent(op), to: MLIR.CAPI, as: :mlirOperationGetParentOperation
  defdelegate destroy(op), to: MLIR.CAPI, as: :mlirOperationDestroy
  defdelegate clone(op), to: MLIR.CAPI, as: :mlirOperationClone
  defdelegate result(op, pos), to: MLIR.CAPI, as: :mlirOperationGetResult

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

  @doc """
  Check if the operation is a terminator.
  """
  def terminator?(%__MODULE__{} = op) do
    MLIR.Context.terminator?(MLIR.context(op), name(op))
  end

  def implements_interface?(%__MODULE__{} = op, interface_id) do
    mlirOperationImplementsInterface(op, interface_id)
    |> Beaver.Native.to_term()
  end

  def infer_type?(%__MODULE__{} = op) do
    implements_interface?(op, mlirInferTypeOpInterfaceTypeID())
  end

  def infer_shaped?(%__MODULE__{} = op) do
    implements_interface?(op, mlirInferShapedTypeOpInterfaceTypeID())
  end
end
