defmodule Beaver.Changeset do
  @moduledoc false
  alias Beaver.MLIR
  require Logger

  @doc """
  Changeset of entities to create an operation.
  """
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

  @type attribute() :: MLIR.Attribute.t() | (MLIR.Context.t() -> MLIR.Attribute.t())
  @type operand() :: MLIR.Value.t() | (MLIR.Context.t() -> MLIR.Value.t())
  @type tagged_operand() :: {atom(), operand() | [operand()]} | operand()
  @type operand_argument() :: tagged_operand() | [tagged_operand()]
  @type type_argument() :: MLIR.Type.t() | (MLIR.Context.t() -> MLIR.Type.t())
  @type tagged_attribute :: {atom(), type_argument() | attribute()}
  @type attribute_argument() :: tagged_attribute() | [tagged_attribute()]
  @type branching_argument() :: MLIR.Block.t() | {MLIR.Block.t(), [MLIR.Value.t()]}
  @type region_argument() :: MLIR.Region.t() | (-> [MLIR.Region.t()])
  @type loc_argument() :: MLIR.Location.t() | {:loc, MLIR.Location.t()}

  @type argument() ::
          operand_argument()
          | attribute_argument()
          | branching_argument()
          | region_argument()
          | loc_argument()

  @spec add_argument(t(), argument()) :: t()

  @doc """
  Add an ODS argument to changeset.
  """

  def add_argument(%__MODULE__{} = changeset, argument) when is_list(argument) do
    for arg <- argument, reduce: changeset do
      changeset -> add_argument(changeset, arg)
    end
  end

  def add_argument(%__MODULE__{context: context} = changeset, {tag, f})
      when is_atom(tag) and is_function(f, 1) do
    add_argument(changeset, {tag, Beaver.Deferred.create(f, context)})
  end

  # f should return regions
  def add_argument(%__MODULE__{} = changeset, f) when is_function(f, 0) do
    add_argument(changeset, f.())
  end

  # If the context is not set, we extract it from the argument
  def add_argument(%__MODULE__{context: nil} = changeset, %{} = argument) do
    add_argument(%__MODULE__{changeset | context: MLIR.context(argument)}, argument)
  end

  def add_argument(%__MODULE__{regions: regions} = changeset, %MLIR.Region{} = region) do
    %__MODULE__{changeset | regions: regions ++ [region]}
  end

  def add_argument(%__MODULE__{} = changeset, {:loc, %Beaver.MLIR.Location{} = location}) do
    %__MODULE__{changeset | location: location}
  end

  def add_argument(
        %__MODULE__{successors: successors, operands: operands} = changeset,
        {%MLIR.Block{} = successor_block, block_args}
      ) do
    %__MODULE__{
      changeset
      | successors: successors ++ [successor_block],
        operands: operands ++ block_args
    }
  end

  def add_argument(
        %__MODULE__{successors: successors} = changeset,
        %MLIR.Block{} = successor
      ) do
    %__MODULE__{changeset | successors: successors ++ [successor]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = changeset,
        {name, :infer}
      )
      when name in [
             :operand_segment_sizes,
             :operandSegmentSizes,
             :result_segment_sizes,
             :resultSegmentSizes
           ] do
    %__MODULE__{changeset | attributes: attributes ++ [{name, :infer}]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = changeset,
        {name, %MLIR.Attribute{} = attr}
      )
      when is_atom(name) do
    %__MODULE__{changeset | attributes: attributes ++ [{name, attr}]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = changeset,
        {name, %MLIR.Type{} = type}
      )
      when is_atom(name) do
    %__MODULE__{changeset | attributes: attributes ++ [{name, type}]}
  end

  def add_argument(
        %__MODULE__{attributes: attributes} = changeset,
        {name, attr}
      )
      when is_atom(name) and is_binary(attr) do
    %__MODULE__{changeset | attributes: attributes ++ [{name, attr}]}
  end

  def add_argument(
        %__MODULE__{operands: operands} = changeset,
        {operand_name, %MLIR.Value{}} = operand
      )
      when is_atom(operand_name) do
    %__MODULE__{changeset | operands: operands ++ [operand]}
  end

  def add_argument(
        %__MODULE__{operands: operands} = changeset,
        {operand_name, variadic_operands} = operand
      )
      when is_atom(operand_name) and is_list(variadic_operands) do
    %__MODULE__{changeset | operands: operands ++ [operand]}
  end

  def add_argument(
        %__MODULE__{operands: operands} = changeset,
        %MLIR.Value{} = operand
      ) do
    %__MODULE__{changeset | operands: operands ++ [operand]}
  end

  def add_argument(%__MODULE__{}, argument) do
    raise ArgumentError, """
    Invalid argument received: #{inspect(argument)}

    Supported argument types:
    - (list of) value
    - (list of) {atom, attribute | type | binary()}
    - block or {block, values} (for block successors)
    - region or (-> [region])
    - location or {:loc, location} (for locations)
    """
  end

  @type type() :: MLIR.Type.t() | [MLIR.Type.t()]
  @type result() :: type() | (MLIR.Context.t() -> type())
  @spec add_result(t(), result()) :: t()

  def add_result(%__MODULE__{} = changeset, argument) when is_list(argument) do
    for arg <- argument, reduce: changeset do
      changeset -> add_result(changeset, arg)
    end
  end

  def add_result(%__MODULE__{context: context} = changeset, f) when is_function(f, 1) do
    add_result(changeset, Beaver.Deferred.create(f, context))
  end

  def add_result(%__MODULE__{results: :infer}, type) when type != :infer do
    raise ArgumentError, "already set to infer the result types"
  end

  def add_result(%__MODULE__{} = changeset, :infer) do
    %__MODULE__{changeset | results: :infer}
  end

  def add_result(%__MODULE__{results: results} = changeset, %MLIR.Type{} = result_type) do
    %__MODULE__{changeset | results: results ++ [result_type]}
  end

  def add_result(%__MODULE__{results: results} = changeset, {:parametric, _, _, _f} = result_type) do
    %__MODULE__{changeset | results: results ++ [result_type]}
  end

  def add_result(%__MODULE__{}, result) do
    raise ArgumentError, """
    Invalid result received: #{inspect(result)}

    Supported results:
    - (list of) type
    - function that returns type
    - :infer
    """
  end

  @doc false
  def reorder_operands(
        %__MODULE__{operands: operands, name: name, attributes: attributes, context: context} =
          changeset
      ) do
    case {should_reorder?(operands), MLIR.ODS.Dump.lookup(name)} do
      {true, {:ok, op_dump}} ->
        reorder_operands_with_dump(changeset, op_dump, attributes, context)

      _ ->
        changeset
    end
  end

  defp reorder_operands_with_dump(
         changeset,
         op_dump,
         attributes,
         %MLIR.Context{} = context
       ) do
    operand_segment_sizes =
      attributes[:operand_segment_sizes] || attributes[:operandSegmentSizes]

    {operands, segment_sizes} = process_operands(changeset.operands, op_dump)

    attributes =
      case operand_segment_sizes do
        :infer ->
          attributes
          |> Keyword.delete(:operand_segment_sizes)
          |> Keyword.delete(:operandSegmentSizes)
          |> Keyword.put(:operand_segment_sizes, Beaver.Deferred.create(segment_sizes, context))

        _ ->
          attributes
      end

    %__MODULE__{changeset | operands: operands, attributes: attributes}
  end

  defp should_reorder?(operands) do
    grouped = Enum.group_by(operands, &match?({_atom, _value}, &1))

    has_tagged = not Enum.empty?(grouped[true] || [])
    has_untagged = not Enum.empty?(grouped[false] || [])

    if has_tagged and has_untagged do
      raise ArgumentError,
            "Cannot mix tagged and untagged operands"
    end

    has_tagged
  end

  defp compare_tag(tag, operand_name) do
    tag = to_string(tag)
    operand_name == tag or Macro.camelize(operand_name) == Macro.camelize(tag)
  end

  defp process_operands(operands, op_dump) do
    # 1. Normalize provided operands for efficient lookup (O(N))
    # This avoids the nested loop by ensuring keys are strings, matching op_dump.
    provided_operands = Map.new(operands)
    op_name = op_dump["name"]

    # 2. Process defined operands in a single pass (O(M log N))
    # A `for` comprehension clearly expresses the transformation.
    tagged_operands =
      for %{"name" => operand_name, "kind" => kind} <- op_dump["operands"] do
        pair = Enum.find(provided_operands, fn {key, _} -> compare_tag(key, operand_name) end)

        # Warn if a required single operand is missing
        if is_nil(pair) and kind == "Single" do
          Logger.warning(
            "Single operand '#{operand_name}' not set when creating operation #{op_name}"
          )
        end

        # Ensure the final value is always a list for consistency.
        # `List.wrap/1` correctly handles both single values and lists.
        # The `|| []` ensures missing operands become an empty list.
        values_as_list = if is_nil(pair), do: [], else: List.wrap(elem(pair, 1))

        {String.to_atom(operand_name), values_as_list}
      end

    # 3. Assemble the final result from the processed list
    # This is cleaner as it operates on a well-structured intermediate variable.
    all_values = Keyword.values(tagged_operands)
    segment_sizes = MLIR.ODS.segment_sizes(Enum.map(all_values, &length/1))
    final_operands = List.flatten(all_values)

    {final_operands, segment_sizes}
  end
end
