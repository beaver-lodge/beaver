alias Beaver.MLIR.{Location, StringRef, Type}
alias Beaver.{Changeset, Deferred, Native}

defmodule Beaver.MLIR.Operation.State do
  @moduledoc false

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Native

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  defp prepare(%Changeset{location: %Location{} = loc, context: nil} = changeset),
    do: %Changeset{changeset | context: MLIR.context(loc)}

  defp prepare(%Changeset{location: nil, context: ctx} = changeset) when not is_nil(ctx),
    do: %Changeset{changeset | location: Location.unknown(ctx: ctx)}

  defp prepare(%Changeset{location: %Deferred{} = loc, context: %MLIR.Context{} = ctx} = changeset),
    do: %Changeset{changeset | location: Deferred.resolve(loc, ctx)}

  defp prepare(%Changeset{location: %Location{} = loc, context: %MLIR.Context{} = ctx} = changeset),
    do: %Changeset{changeset | location: Deferred.resolve(loc, ctx)}

  defp add_attributes(state, []), do: state

  defp add_attributes(%__MODULE__{} = state, attr_kw)
       when is_list(attr_kw) do
    ctx = CAPI.beaverOperationStateGetContext(state)

    attr_list =
      for {k, v} <- attr_kw do
        attr =
          case v do
            v when is_binary(v) -> CAPI.mlirAttributeParseGet(ctx, StringRef.create(v))
            %Type{} = type -> Beaver.MLIR.Attribute.type(type)
            %Beaver.MLIR.Attribute{} -> v
            _ -> raise "attribute not supported: #{inspect({k, v})}"
          end

        if MLIR.null?(attr), do: raise("attribute can't be null, #{inspect({k, v})}")

        CAPI.mlirNamedAttributeGet(CAPI.mlirIdentifierGet(ctx, StringRef.create(Atom.to_string(k))), attr)
      end

    add_array(state, attr_list, MLIR.NamedAttribute, :mlirOperationStateAddAttributes)
  end

  defp add_array(state, values, kind, function) do
    if values not in [[], nil],
      do: apply(CAPI, function, [Native.ptr(state), length(values), Native.array(values, kind)])

    state
  end

  defp add_results(%__MODULE__{} = state, []), do: state

  defp add_results(%__MODULE__{} = state, :infer),
    do: state |> tap(&CAPI.mlirOperationStateEnableResultTypeInference(Native.ptr(&1)))

  defp add_results(%__MODULE__{} = state, types)
       when is_list(types) do
    context = CAPI.beaverOperationStateGetContext(state)

    array =
      types
      |> Enum.map(&Deferred.resolve(&1, context))
      |> Enum.map(fn
        t when is_binary(t) -> CAPI.mlirTypeParseGet(context, StringRef.create(t))
        %Type{} = t -> t
      end)
      |> Native.array(Type)

    CAPI.mlirOperationStateAddResults(Native.ptr(state), length(types), array)
    state
  end

  @doc """
  Create a new operation state in MLIR CAPI.
  """
  def create(%Changeset{} = changeset) do
    changeset = changeset |> Changeset.reorder_operands() |> prepare()

    changeset.name
    |> StringRef.create()
    |> CAPI.mlirOperationStateGet(changeset.location)
    |> add_attributes(changeset.attributes)
    |> add_array(changeset.operands, MLIR.Value, :mlirOperationStateAddOperands)
    |> add_array(changeset.successors, MLIR.Block, :mlirOperationStateAddSuccessors)
    |> add_array(changeset.regions, MLIR.Region, :mlirOperationStateAddOwnedRegions)
    |> add_results(changeset.results)
  end
end
