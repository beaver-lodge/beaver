alias Beaver.MLIR.{Location, Operation, StringRef, Type}
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

  defp add_attributes(%Operation.State{} = state, attr_kw)
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

    CAPI.mlirOperationStateAddAttributes(
      Native.ptr(state),
      length(attr_list),
      Native.array(attr_list, MLIR.NamedAttribute)
    )

    state
  end

  defp add_operands(%Operation.State{} = state, []), do: state

  defp add_operands(%Operation.State{} = state, operands) do
    array = operands |> Native.array(MLIR.Value)

    CAPI.mlirOperationStateAddOperands(Native.ptr(state), length(operands), array)
    state
  end

  defp add_results(%Operation.State{} = state, []), do: state

  defp add_results(%Operation.State{} = state, :infer),
    do: state |> tap(&CAPI.mlirOperationStateEnableResultTypeInference(Native.ptr(&1)))

  defp add_results(%Operation.State{} = state, types)
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

  defp add_regions(%Operation.State{} = state, empty) when empty in [[], nil], do: state

  defp add_regions(%Operation.State{} = state, regions) when is_list(regions) do
    Enum.each(regions, fn
      %MLIR.Region{} -> :ok
      other -> raise "not a region: #{inspect(other)}"
    end)

    array_ptr = regions |> Native.array(MLIR.Region)

    CAPI.mlirOperationStateAddOwnedRegions(Native.ptr(state), length(regions), array_ptr)

    state
  end

  defp add_successors(state, []), do: state

  defp add_successors(%Operation.State{} = state, successors)
       when is_list(successors) do
    array_ptr = successors |> Native.array(MLIR.Block)

    CAPI.mlirOperationStateAddSuccessors(Native.ptr(state), length(successors), array_ptr)

    state
  end

  @doc """
  Create a new operation state in MLIR CAPI.
  """
  def create(%Changeset{} = changeset) do
    %Changeset{
      name: name,
      attributes: attributes,
      operands: operands,
      results: results,
      successors: successors,
      regions: regions,
      location: location,
      context: _context
    } = changeset |> Changeset.reorder_operands() |> prepare()

    name
    |> StringRef.create()
    |> CAPI.mlirOperationStateGet(location)
    |> add_attributes(attributes)
    |> add_operands(operands)
    |> add_successors(successors)
    |> add_regions(regions)
    |> add_results(results)
  end
end
