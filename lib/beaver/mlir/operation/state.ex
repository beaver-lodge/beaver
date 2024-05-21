defmodule Beaver.MLIR.Operation.State do
  @moduledoc false

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  defp prepare(
         %MLIR.Operation.Changeset{
           location: location,
           context: nil
         } = changeset
       )
       when not is_nil(location) and not is_function(location) do
    %MLIR.Operation.Changeset{changeset | context: CAPI.mlirLocationGetContext(location)}
  end

  defp prepare(
         %MLIR.Operation.Changeset{
           location: nil,
           context: context
         } = changeset
       )
       when not is_nil(context) do
    %MLIR.Operation.Changeset{changeset | location: CAPI.mlirLocationUnknownGet(context)}
  end

  defp prepare(
         %MLIR.Operation.Changeset{
           location: location,
           context: context
         } = changeset
       )
       when not is_nil(context) and is_function(location, 1) do
    %MLIR.Operation.Changeset{changeset | location: location.(context)}
  end

  defp prepare(
         %MLIR.Operation.Changeset{
           location: %Beaver.MLIR.Location{} = location
         } = changeset
       ) do
    %MLIR.Operation.Changeset{changeset | location: location}
  end

  defp add_attributes(state, []) do
    state
  end

  defp add_attributes(%MLIR.Operation.State{} = state, attr_kw)
       when is_list(attr_kw) do
    ctx = CAPI.beaverMlirOperationStateGetContext(state)

    attr_list =
      for {k, v} <- attr_kw do
        attr =
          case v do
            v when is_binary(v) ->
              CAPI.mlirAttributeParseGet(ctx, MLIR.StringRef.create(v))

            %Beaver.MLIR.Type{} = type ->
              Beaver.MLIR.Attribute.type(type)

            %Beaver.MLIR.Attribute{} ->
              v

            _ ->
              raise "attribute not supported: #{inspect({k, v})}"
          end

        if MLIR.is_null(attr) do
          raise "attribute can't be null, #{inspect({k, v})}"
        end

        CAPI.mlirNamedAttributeGet(
          CAPI.mlirIdentifierGet(
            ctx,
            MLIR.StringRef.create(Atom.to_string(k))
          ),
          attr
        )
      end

    CAPI.mlirOperationStateAddAttributes(
      Beaver.Native.ptr(state),
      length(attr_list),
      Beaver.Native.array(attr_list, MLIR.NamedAttribute)
    )

    state
  end

  defp add_operands(%MLIR.Operation.State{} = state, []) do
    state
  end

  defp add_operands(%MLIR.Operation.State{} = state, operands) do
    array = operands |> Beaver.Native.array(MLIR.Value)

    CAPI.mlirOperationStateAddOperands(Beaver.Native.ptr(state), length(operands), array)
    state
  end

  defp add_results(%MLIR.Operation.State{} = state, []) do
    state
  end

  defp add_results(%MLIR.Operation.State{} = state, result_types)
       when is_list(result_types) do
    context = CAPI.beaverMlirOperationStateGetContext(state)

    array =
      result_types
      |> Enum.map(&Beaver.Deferred.create(&1, context))
      |> Enum.map(fn
        t when is_binary(t) ->
          CAPI.mlirTypeParseGet(context, MLIR.StringRef.create(t))

        %MLIR.Type{} = t ->
          t
      end)
      |> Beaver.Native.array(MLIR.Type)

    CAPI.mlirOperationStateAddResults(Beaver.Native.ptr(state), length(result_types), array)
    state
  end

  defp add_regions(%MLIR.Operation.State{} = state, empty) when empty in [[], nil] do
    state
  end

  defp add_regions(%MLIR.Operation.State{} = state, regions) when is_list(regions) do
    Enum.each(regions, fn
      %MLIR.Region{} ->
        :ok

      other ->
        raise "not a region: #{inspect(other)}"
    end)

    array_ptr = regions |> Beaver.Native.array(MLIR.Region)

    CAPI.mlirOperationStateAddOwnedRegions(
      Beaver.Native.ptr(state),
      length(regions),
      array_ptr
    )

    state
  end

  defp add_successors(state, []) do
    state
  end

  defp add_successors(%MLIR.Operation.State{} = state, successors)
       when is_list(successors) do
    array_ptr = successors |> Beaver.Native.array(MLIR.Block)

    CAPI.mlirOperationStateAddSuccessors(
      Beaver.Native.ptr(state),
      length(successors),
      array_ptr
    )

    state
  end

  @doc """
  Create a new operation state in MLIR CAPI.
  """
  def create(%MLIR.Operation.Changeset{} = changeset) do
    %MLIR.Operation.Changeset{
      name: name,
      attributes: attributes,
      operands: operands,
      results: results,
      successors: successors,
      regions: regions,
      location: location,
      context: _context
    } = prepare(changeset)

    name
    |> MLIR.StringRef.create()
    |> CAPI.mlirOperationStateGet(location)
    |> add_attributes(attributes)
    |> add_operands(operands)
    |> add_successors(successors)
    |> add_regions(regions)
    |> add_results(results)
  end
end
