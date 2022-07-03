defmodule Beaver.MLIR.Operation.State do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  alias Exotic.Value.Array
  alias Exotic.Value
  defstruct [:ref]

  def get!(name, location) when is_binary(name) do
    CAPI.mlirOperationStateGet(
      MLIR.StringRef.create(name),
      location
    )
  end

  def get!(context, name) when is_binary(name) do
    get!(name, CAPI.mlirLocationUnknownGet(context))
  end

  defp get_context(state) do
    location = Exotic.Value.fetch(state, CAPI.MlirOperationState, :location)
    CAPI.mlirLocationGetContext(location)
  end

  def add_attr(state, attrs) when is_list(attrs) do
    ctx = get_context(state)

    named_attrs =
      for {k, v} <- attrs do
        k = Atom.to_string(k)

        attr =
          case v do
            v when is_binary(v) ->
              CAPI.mlirAttributeParseGet(ctx, MLIR.StringRef.create(v))

            %Beaver.MLIR.CAPI.MlirType{} = type ->
              Beaver.MLIR.Attribute.type(type)

            _ ->
              v
          end
          |> Exotic.Value.transmit()

        CAPI.mlirNamedAttributeGet(
          CAPI.mlirIdentifierGet(ctx, MLIR.StringRef.create(k)),
          attr
        )
      end

    array_ptr = named_attrs |> Enum.map(&Value.transmit/1) |> Array.get() |> Value.get_ptr()

    CAPI.mlirOperationStateAddAttributes(
      Value.get_ptr(state),
      length(attrs),
      array_ptr
    )

    state
  end

  def add_operand(state, operands) do
    array =
      operands
      |> Enum.map(&Exotic.Value.transmit/1)
      |> Array.get()
      |> Value.get_ptr()

    CAPI.mlirOperationStateAddOperands(Value.get_ptr(state), length(operands), array)
    state
  end

  def add_result(state, result_types) do
    context = get_context(state)

    array =
      result_types
      |> Enum.map(fn
        t when is_binary(t) ->
          CAPI.mlirTypeParseGet(context, MLIR.StringRef.create(t))

        t ->
          t
      end)
      |> Enum.map(&Exotic.Value.transmit/1)
      |> Array.get()
      |> Value.get_ptr()

    CAPI.mlirOperationStateAddResults(Exotic.Value.get_ptr(state), length(result_types), array)
    state
  end

  def add_regions(state, regions) when is_list(regions) do
    array_ptr = regions |> Enum.map(&Value.transmit/1) |> Array.get() |> Value.get_ptr()

    CAPI.mlirOperationStateAddOwnedRegions(
      Value.get_ptr(state),
      length(regions),
      array_ptr
    )

    state
  end

  def add_successors(state, successors) when is_list(successors) do
    array_ptr = successors |> Enum.map(&Value.transmit/1) |> Array.get() |> Value.get_ptr()

    CAPI.mlirOperationStateAddSuccessors(
      Exotic.Value.get_ptr(state),
      length(successors),
      array_ptr
    )

    state
  end

  @doc """
  Add an ODS argument to state. It the could one of the following:
  - An value/result as an operand
  - {name, attribute} pair as a named attribute
  - {:regions, fn/0} a function to create regions
  - {:result_types, types} as the return types of the operation
  - {:successor, block} a successor block
  """
  def add_argument(state, {:defer_if_terminator, _}) do
    state
  end

  def add_argument(state, value) when is_integer(value) do
    add_attr(state, value: "#{value}")
  end

  def add_argument(state, {:regions, region_filler}) when is_function(region_filler, 0) do
    regions = region_filler.()
    add_regions(state, regions)
  end

  def add_argument(_state, {:regions, _}) do
    raise "the function to create regions shuold have a arity of 0"
  end

  def add_argument(state, {:result_types, result_types}) when is_list(result_types) do
    add_result(state, result_types)
  end

  def add_argument(state, {:result_types, result_types}) do
    add_result(state, [result_types])
  end

  def add_argument(state, {:successor, successor}) when is_atom(successor) do
    successor_block = MLIR.Managed.Terminator.get_block(successor)

    if is_nil(successor_block), do: raise("successor block not found: #{successor}")

    state
    |> add_successors([successor_block])
  end

  def add_argument(state, {name, attr}) do
    add_attr(state, [{name, attr}])
  end

  def add_argument(
        state,
        operand = %Beaver.MLIR.CAPI.MlirValue{}
      ) do
    add_operand(state, [operand])
  end

  def add_argument(
        state,
        operand = %Beaver.MLIR.CAPI.MlirValue{}
      ) do
    add_operand(state, [operand])
  end
end
