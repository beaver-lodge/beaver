defmodule Beaver.MLIR.Operation.State do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI
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
    location = CAPI.beaverMlirOperationStateGetLocation(state)
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

        CAPI.mlirNamedAttributeGet(
          CAPI.mlirIdentifierGet(ctx, MLIR.StringRef.create(k)),
          attr
        )
      end

    array_ptr = named_attrs |> CAPI.array()

    CAPI.mlirOperationStateAddAttributes(
      CAPI.ptr(state),
      length(attrs),
      array_ptr
    )

    state
  end

  def add_operand(state, operands) do
    array = CAPI.array(operands)

    CAPI.mlirOperationStateAddOperands(CAPI.ptr(state), length(operands), array)
    state
  end

  def add_result(state, []) do
    CAPI.mlirOperationStateAddResults(CAPI.ptr(state), 0, Exotic.Value.Ptr.null())
    state
  end

  def add_result(state, result_types) when is_list(result_types) do
    context = get_context(state)

    array =
      result_types
      |> Enum.map(fn
        t when is_binary(t) ->
          CAPI.mlirTypeParseGet(context, MLIR.StringRef.create(t))

        t ->
          t
      end)
      |> CAPI.array()

    CAPI.mlirOperationStateAddResults(CAPI.ptr(state), length(result_types), array)
    state
  end

  def add_regions(state, regions) when is_list(regions) do
    Enum.each(regions, fn
      %CAPI.MlirRegion{} ->
        :ok

      other ->
        raise "not a region: #{inspect(other)}"
    end)

    array_ptr = regions |> CAPI.array()

    CAPI.mlirOperationStateAddOwnedRegions(
      CAPI.ptr(state),
      length(regions),
      array_ptr
    )

    state
  end

  def add_successors(state, successors) when is_list(successors) do
    array_ptr = successors |> CAPI.array()

    CAPI.mlirOperationStateAddSuccessors(
      CAPI.ptr(state),
      length(successors),
      array_ptr
    )

    state
  end

  @doc """
  Add an ODS argument to state.
  """

  @type argument() ::
          CAPI.MlirValue.t()
          | {atom(), CAPI.MlirAttribute.t()}
          | {:regions, function()}
          | {:result_types, [CAPI.MlirType.t()]}
          | {:successor, CAPI.MlirBlock.t()}
  @spec add_argument(CAPI.MlirOperationState.t(), argument()) :: CAPI.MlirOperationState.t()

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

  def add_argument(state, {:result_types, %MLIR.CAPI.MlirType{} = result_type}) do
    add_result(state, [result_type])
  end

  def add_argument(state, {:successor, %Beaver.MLIR.CAPI.MlirBlock{} = successor_block}) do
    state
    |> add_successors([successor_block])
  end

  def add_argument(state, {:successor, successor}) when is_atom(successor) do
    successor_block = MLIR.Managed.Terminator.get_block(successor)

    if is_nil(successor_block), do: raise("successor block not found: #{successor}")

    state
    |> add_successors([successor_block])
  end

  def add_argument(state, {name, %MLIR.CAPI.MlirAttribute{} = attr}) when is_atom(name) do
    add_attr(state, [{name, attr}])
  end

  def add_argument(state, {name, %MLIR.CAPI.MlirType{} = type_attr}) when is_atom(name) do
    add_attr(state, [{name, type_attr}])
  end

  def add_argument(state, {name, attr}) when is_atom(name) and is_binary(attr) do
    add_attr(state, [{name, attr}])
  end

  def add_argument(state, %Beaver.MLIR.CAPI.MlirValue{} = operand) do
    add_operand(state, [operand])
  end
end
