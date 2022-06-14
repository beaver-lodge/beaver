defmodule Beaver.MLIR.Operation.State do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR.CAPI.IR
  alias Exotic.Value.Array
  alias Exotic.Value
  defstruct [:ref]

  def get!(name, location) when is_binary(name) do
    IR.mlirOperationStateGet(
      IR.string_ref(name),
      location
    )
  end

  def get!(context, name) when is_binary(name) do
    get!(name, IR.mlirLocationUnknownGet(context))
  end

  defp get_context(state) do
    location = Exotic.Value.fetch(state, IR.OperationState, :location)
    IR.mlirLocationGetContext(location)
  end

  def add_attr(state, attrs) when is_list(attrs) do
    ctx = get_context(state)

    named_attrs =
      for {k, v} <- attrs do
        k = Atom.to_string(k)

        attr =
          case v do
            v when is_binary(v) ->
              IR.mlirAttributeParseGet(ctx, IR.string_ref(v))

            _ ->
              v
          end
          |> Exotic.Value.transmit()

        IR.mlirNamedAttributeGet(
          IR.mlirIdentifierGet(ctx, IR.string_ref(k)),
          attr
        )
      end

    array_ptr = named_attrs |> Enum.map(&Value.transmit/1) |> Array.get() |> Value.get_ptr()

    IR.mlirOperationStateAddAttributes(
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

    IR.mlirOperationStateAddOperands(Value.get_ptr(state), length(operands), array)
    state
  end

  def add_result(state, result_types) do
    context = get_context(state)

    array =
      result_types
      |> Enum.map(fn
        t when is_binary(t) ->
          IR.mlirTypeParseGet(context, IR.string_ref(t))

        t ->
          t
      end)
      |> Enum.map(&Exotic.Value.transmit/1)
      |> Array.get()
      |> Value.get_ptr()

    IR.mlirOperationStateAddResults(Exotic.Value.get_ptr(state), length(result_types), array)
    state
  end

  def add_regions(state, regions) when is_list(regions) do
    array_ptr = regions |> Enum.map(&Value.transmit/1) |> Array.get() |> Value.get_ptr()

    IR.mlirOperationStateAddOwnedRegions(
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
        operand = %Exotic.Value{type: %Exotic.Type{t: Beaver.MLIR.CAPI.IR.Value}}
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
