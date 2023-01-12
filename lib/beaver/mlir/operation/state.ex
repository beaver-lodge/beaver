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
          location: MLIR.Location.t(),
          context: MLIR.Context.t()
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

  defp add_attributes(%MLIR.CAPI.MlirOperationState{} = state, attr_kw)
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
          end

        if MLIR.is_null(attr) do
          raise "attribute can't be null, #{inspect({k, v})}"
        end

        attr
      end

    name_list = Enum.map(attr_kw, fn {k, _} -> Atom.to_string(k) |> MLIR.StringRef.create() end)

    CAPI.beaverOperationStateAddAttributes(
      ctx,
      Beaver.Native.ptr(state),
      length(attr_kw),
      Beaver.Native.array(name_list, MLIR.StringRef),
      Beaver.Native.array(attr_list, MLIR.Attribute)
    )

    state
  end

  defp add_operands(%MLIR.CAPI.MlirOperationState{} = state, []) do
    state
  end

  defp add_operands(%MLIR.CAPI.MlirOperationState{} = state, operands) do
    array = operands |> Beaver.Native.array(MLIR.Value)

    CAPI.mlirOperationStateAddOperands(Beaver.Native.ptr(state), length(operands), array)
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

  defp add_regions(%MLIR.CAPI.MlirOperationState{} = state, []) do
    state
  end

  defp add_regions(%MLIR.CAPI.MlirOperationState{} = state, regions) when is_list(regions) do
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

  defp add_successors(%MLIR.CAPI.MlirOperationState{} = state, successors)
       when is_list(successors) do
    array_ptr = successors |> Beaver.Native.array(MLIR.Block)

    CAPI.mlirOperationStateAddSuccessors(
      Beaver.Native.ptr(state),
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
       when not is_nil(location) and not is_function(location) do
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

  defp prepare(
         %__MODULE__{
           location: location,
           context: context
         } = state
       )
       when not is_nil(context) and is_function(location, 1) do
    %{state | location: location.(context)}
  end

  defp prepare(
         %__MODULE__{
           location: %Beaver.MLIR.Location{} = location
         } = state
       ) do
    %{state | location: location}
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
          | {:result_types, [MLIR.Type.t()]}
          | {MLIR.Block.t(), [MLIR.Value.t()]}
          | MLIR.Block.t()
  @spec add_argument(t(), argument()) :: CAPI.MlirOperationState.t()

  @doc """
  Add an ODS argument to state.
  """

  def add_argument(%__MODULE__{context: context} = state, {tag, f})
      when is_atom(tag) and is_function(f, 1) do
    add_argument(state, {tag, f.(context)})
  end

  def add_argument(%__MODULE__{context: context} = state, f) when is_function(f, 1) do
    add_argument(state, f.(context))
  end

  def add_argument(%__MODULE__{regions: regions} = state, %MLIR.Region{} = region) do
    %{state | regions: regions ++ [region]}
  end

  def add_argument(%__MODULE__{} = state, {:loc, %Beaver.MLIR.Location{} = location}) do
    %{state | location: location}
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

  def add_argument(%__MODULE__{results: results} = state, %MLIR.Type{} = result_type) do
    %{state | results: results ++ [result_type]}
  end

  def add_argument(
        %__MODULE__{successors: successors, operands: operands} = state,
        {%Beaver.MLIR.Block{} = successor_block, block_args}
      ) do
    %{state | successors: successors ++ [successor_block], operands: operands ++ block_args}
  end

  def add_argument(
        %__MODULE__{successors: successors} = state,
        %MLIR.Block{} = successor
      ) do
    %{state | successors: successors ++ [successor]}
  end

  def add_argument(%__MODULE__{attributes: attributes} = state, value) when is_integer(value) do
    %{state | attributes: attributes ++ [value: value]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = state,
        {name, %MLIR.Attribute{} = attr}
      )
      when is_atom(name) do
    %{state | attributes: attributes ++ [{name, attr}]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = state,
        {name, %MLIR.Type{} = type}
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
