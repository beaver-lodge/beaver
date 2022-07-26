defmodule Beaver.MLIR.Operation.State do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  @type t() :: %__MODULE__{
          operands: list(),
          results: list(),
          successors: list(),
          attributes: list(),
          name: String.t(),
          regions: list(),
          location: MLIR.CAPI.MlirLocation.t(),
          context: MLIR.CAPI.MlirContext.t()
        }
  defstruct operands: [],
            results: [],
            successors: [],
            attributes: [],
            name: nil,
            regions: [],
            location: nil,
            context: nil

  defp add_attributes(state, []) do
    state
  end

  # defp add_attributes(%MLIR.CAPI.MlirOperationState{} = state, attr_kw)
  #      when is_list(attr_kw) do
  #   ctx = CAPI.beaverMlirOperationStateGetContext(state)

  #   named_list =
  #     for {k, v} <- attr_kw do
  #       attr =
  #         case v do
  #           v when is_binary(v) ->
  #             CAPI.mlirAttributeParseGet(ctx, MLIR.StringRef.create(v))

  #           %Beaver.MLIR.CAPI.MlirType{} = type ->
  #             Beaver.MLIR.Attribute.type(type)

  #           %Beaver.MLIR.CAPI.MlirAttribute{} ->
  #             v
  #         end

  #       if MLIR.is_null(attr) do
  #         raise "attribute can't be null, #{inspect({k, v})}"
  #       end

  #       CAPI.mlirIdentifierGet(ctx, MLIR.StringRef.create(k))
  #       |> CAPI.mlirNamedAttributeGet(attr)
  #     end
  #     |> CAPI.MlirNamedAttribute.array()

  #   state
  #   |> CAPI.ptr()
  #   |> CAPI.mlirOperationStateAddAttributes(
  #     length(attr_kw),
  #     named_list
  #   )

  #   state |> CAPI.bag(named_list)
  # end

  defp add_attributes(%MLIR.CAPI.MlirOperationState{} = state, attr_kw)
       when is_list(attr_kw) do
    ctx = CAPI.beaverMlirOperationStateGetContext(state)

    attr_list =
      for {k, v} <- attr_kw do
        attr =
          case v do
            v when is_binary(v) ->
              CAPI.mlirAttributeParseGet(ctx, MLIR.StringRef.create(v))

            %Beaver.MLIR.CAPI.MlirType{} = type ->
              Beaver.MLIR.Attribute.type(type)

            %Beaver.MLIR.CAPI.MlirAttribute{} ->
              v
          end

        if MLIR.is_null(attr) do
          raise "attribute can't be null, #{inspect({k, v})}"
        end

        attr
      end

    name_list = Enum.map(attr_kw, fn {k, _} -> Atom.to_string(k) |> MLIR.StringRef.create() end)

    CAPI.beaverOperationStateAddAttributes(
      ctx,
      CAPI.ptr(state),
      length(attr_kw),
      CAPI.MlirStringRef.array(name_list),
      CAPI.MlirAttribute.array(attr_list)
    )

    state
  end

  defp add_operands(%MLIR.CAPI.MlirOperationState{} = state, []) do
    state
  end

  defp add_operands(%MLIR.CAPI.MlirOperationState{} = state, operands) do
    array = operands |> MLIR.Value.array()

    CAPI.mlirOperationStateAddOperands(CAPI.ptr(state), length(operands), array)
    state
  end

  defp add_results(%MLIR.CAPI.MlirOperationState{} = state, []) do
    state
  end

  defp add_results(%MLIR.CAPI.MlirOperationState{} = state, result_types)
       when is_list(result_types) do
    context = CAPI.beaverMlirOperationStateGetContext(state)

    array =
      result_types
      |> Enum.map(fn
        t when is_binary(t) ->
          CAPI.mlirTypeParseGet(context, MLIR.StringRef.create(t))

        %CAPI.MlirType{} = t ->
          t
      end)
      |> CAPI.MlirType.array()

    CAPI.mlirOperationStateAddResults(CAPI.ptr(state), length(result_types), array)
    state
  end

  defp add_regions(%MLIR.CAPI.MlirOperationState{} = state, []) do
    state
  end

  defp add_regions(%MLIR.CAPI.MlirOperationState{} = state, regions) when is_list(regions) do
    Enum.each(regions, fn
      %CAPI.MlirRegion{} ->
        :ok

      other ->
        raise "not a region: #{inspect(other)}"
    end)

    array_ptr = regions |> CAPI.MlirRegion.array()

    CAPI.mlirOperationStateAddOwnedRegions(
      CAPI.ptr(state),
      length(regions),
      array_ptr
    )

    state
  end

  defp add_successors(state, []) do
    state
  end

  defp add_successors(%MLIR.CAPI.MlirOperationState{} = state, successors)
       when is_list(successors) do
    array_ptr = successors |> CAPI.MlirBlock.array()

    CAPI.mlirOperationStateAddSuccessors(
      CAPI.ptr(state),
      length(successors),
      array_ptr
    )

    state
  end

  defp prepare(
         %__MODULE__{
           location: location,
           context: nil
         } = state
       )
       when not is_nil(location) do
    %{state | context: CAPI.mlirLocationGetContext(location)}
  end

  defp prepare(
         %__MODULE__{
           location: nil,
           context: context
         } = state
       )
       when not is_nil(context) do
    %{state | location: CAPI.mlirLocationUnknownGet(context)}
  end

  @doc """
  Create a new operation state in MLIR CAPI.
  """
  def create(%__MODULE__{} = state) do
    state = prepare(state)

    %__MODULE__{
      name: name,
      attributes: attributes,
      operands: operands,
      results: results,
      successors: successors,
      regions: regions,
      location: location,
      context: _context
    } = state

    name
    |> MLIR.StringRef.create()
    |> CAPI.mlirOperationStateGet(location)
    |> add_attributes(attributes)
    |> add_operands(operands)
    |> add_successors(successors)
    |> add_regions(regions)
    |> add_results(results)
  end

  @type argument() ::
          Value.t()
          | {atom(), CAPI.MlirAttribute.t()}
          | {:regions, function()}
          | {:result_types, [CAPI.MlirType.t()]}
          | {:successor, CAPI.MlirBlock.t()}
  @spec add_argument(t(), argument()) :: CAPI.MlirOperationState.t()

  @doc """
  Add an ODS argument to state.
  """
  def add_argument(%__MODULE__{attributes: attributes} = state, value) when is_integer(value) do
    %{state | attributes: attributes ++ [value: value]}
  end

  def add_argument(%__MODULE__{regions: regions} = state, %CAPI.MlirRegion{} = region) do
    %{state | regions: regions ++ [region]}
  end

  def add_argument(%__MODULE__{regions: regions} = state, {:regions, region_filler})
      when is_function(region_filler, 0) do
    %{state | regions: regions ++ region_filler.()}
  end

  def add_argument(_state, {:regions, _}) do
    raise "the function to create regions shuold have a arity of 0"
  end

  def add_argument(%__MODULE__{results: results} = state, {:result_types, result_types})
      when is_list(result_types) do
    %{state | results: results ++ result_types}
  end

  def add_argument(%__MODULE__{results: results} = state, %MLIR.CAPI.MlirType{} = result_type) do
    %{state | results: results ++ [result_type]}
  end

  def add_argument(
        %__MODULE__{successors: successors} = state,
        {:successor, %Beaver.MLIR.CAPI.MlirBlock{} = successor_block}
      ) do
    %{state | successors: successors ++ [successor_block]}
  end

  def add_argument(%__MODULE__{successors: successors} = state, {:successor, successor})
      when is_atom(successor) do
    successor_block = MLIR.Managed.Terminator.get_block(successor)

    if is_nil(successor_block), do: raise("successor block not found: #{successor}")
    %{state | successors: successors ++ [successor_block]}
  end

  def add_argument(
        %__MODULE__{successors: successors} = state,
        %MLIR.CAPI.MlirBlock{} = successor
      ) do
    %{state | successors: successors ++ [successor]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = state,
        {name, %MLIR.CAPI.MlirAttribute{} = attr}
      )
      when is_atom(name) do
    %{state | attributes: attributes ++ [{name, attr}]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = state,
        {name, %MLIR.CAPI.MlirType{} = type}
      )
      when is_atom(name) do
    %{state | attributes: attributes ++ [{name, type}]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = state,
        {name, attr}
      )
      when is_atom(name) and is_binary(attr) do
    %{state | attributes: attributes ++ [{name, attr}]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = state,
        [{name, _attr} | _tail] = attrs
      )
      when is_atom(name) do
    %{state | attributes: attributes ++ attrs}
  end

  def add_argument(
        %__MODULE__{operands: operands} = state,
        %Beaver.MLIR.Value{} = operand
      ) do
    %{state | operands: operands ++ [operand]}
  end
end
