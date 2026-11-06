alias Beaver.MLIR.{Block, Location, ODS.Dump, Region, Type, Value}

defmodule Beaver.Changeset do
  @moduledoc false
  alias Beaver.Deferred
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
          location: Location.t(),
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

  @type attribute() :: Deferred.contextual(MLIR.Attribute.t())
  @type operand() :: Value.t()
  @type tagged_operand() :: {atom(), operand() | [operand()]}
  @type operand_argument() :: tagged_operand() | operand() | [operand()]
  @type type_argument() :: Deferred.contextual(Type.t())
  @type tagged_attribute :: {atom(), type_argument() | attribute() | String.t() | :infer}
  @type attribute_argument() :: tagged_attribute() | [tagged_attribute()]
  @type branching_argument() :: Block.t() | {Block.t(), [Value.t()]}
  @type region_argument() :: Region.t() | (-> [Region.t()])
  @type loc_argument() :: Location.t() | {:loc, Location.t()}

  @type argument() ::
          operand_argument() | attribute_argument() | branching_argument() | region_argument() | loc_argument()

  @spec add_argument(t(), argument()) :: t()

  @doc """
  Add an ODS argument to changeset.
  """

  def add_argument(%__MODULE__{} = changeset, argument) when is_list(argument) do
    for arg <- argument, reduce: changeset do
      changeset -> add_argument(changeset, arg)
    end
  end

  def add_argument(%__MODULE__{context: ctx} = changeset, {tag, %Deferred{} = deferred})
      when is_atom(tag), do: add_argument(changeset, {tag, Deferred.resolve(deferred, ctx)})

  # f should return regions
  def add_argument(%__MODULE__{} = changeset, f) when is_function(f, 0),
    do: add_argument(changeset, f.())

  # If the context is not set, we extract it from the argument
  def add_argument(%__MODULE__{context: nil} = changeset, %{} = argument),
    do: add_argument(%__MODULE__{changeset | context: MLIR.context(argument)}, argument)

  def add_argument(%__MODULE__{regions: regions} = changeset, %Region{} = region),
    do: %__MODULE__{changeset | regions: regions ++ [region]}

  def add_argument(%__MODULE__{context: ctx} = changeset, {:loc, %Location{} = loc}),
    do: %__MODULE__{changeset | location: Deferred.resolve(loc, ctx)}

  def add_argument(
        %__MODULE__{context: ctx, successors: successors, operands: operands} = changeset,
        {%Block{} = successor_block, block_args}
      ) do
    block_args = Enum.map(block_args, &Deferred.resolve(&1, ctx))

    %__MODULE__{changeset | successors: successors ++ [successor_block], operands: operands ++ block_args}
  end

  def add_argument(%__MODULE__{successors: successors} = changeset, %Block{} = successor),
    do: %__MODULE__{changeset | successors: successors ++ [successor]}

  def add_argument(%__MODULE__{attributes: attributes} = changeset, {name, :infer})
      when name in ~w(operand_segment_sizes operandSegmentSizes result_segment_sizes resultSegmentSizes)a,
      do: %__MODULE__{changeset | attributes: attributes ++ [{name, :infer}]}

  def add_argument(%__MODULE__{context: ctx, attributes: attributes} = changeset, {name, %kind{} = value})
      when is_atom(name) and kind in [MLIR.Attribute, Type],
      do: %__MODULE__{changeset | attributes: attributes ++ [{name, Deferred.resolve(value, ctx)}]}

  def add_argument(%__MODULE__{attributes: attributes} = changeset, {name, attr})
      when is_atom(name) and is_binary(attr), do: %__MODULE__{changeset | attributes: attributes ++ [{name, attr}]}

  def add_argument(%__MODULE__{context: ctx, operands: operands} = changeset, {operand_name, %Value{} = value})
      when is_atom(operand_name) do
    operand = {operand_name, Deferred.resolve(value, ctx)}
    %__MODULE__{changeset | operands: operands ++ [operand]}
  end

  def add_argument(%__MODULE__{context: ctx, operands: operands} = changeset, {operand_name, variadic_operands})
      when is_atom(operand_name) and is_list(variadic_operands) do
    operand = {operand_name, Enum.map(variadic_operands, &Deferred.resolve(&1, ctx))}
    %__MODULE__{changeset | operands: operands ++ [operand]}
  end

  def add_argument(%__MODULE__{context: ctx, operands: operands} = changeset, %Value{} = operand) do
    operand = Deferred.resolve(operand, ctx)
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

  @type deferred_type() :: {:parametric, term(), term(), term()}
  @type type() :: Type.t() | deferred_type() | [Type.t() | deferred_type()]
  @type result() :: :infer | type() | Deferred.contextual(type())
  @spec add_result(t(), result()) :: t()

  def add_result(%__MODULE__{} = changeset, argument) when is_list(argument) do
    for arg <- argument, reduce: changeset do
      changeset -> add_result(changeset, arg)
    end
  end

  def add_result(%__MODULE__{context: ctx} = changeset, %Deferred{} = deferred),
    do: add_result(changeset, Deferred.resolve(deferred, ctx))

  def add_result(%__MODULE__{results: :infer}, type) when type != :infer,
    do: raise(ArgumentError, "already set to infer the result types")

  def add_result(%__MODULE__{} = changeset, :infer), do: %__MODULE__{changeset | results: :infer}

  def add_result(%__MODULE__{context: ctx, results: results} = changeset, %Type{} = result_type) do
    result_type = Deferred.resolve(result_type, ctx)
    %__MODULE__{changeset | results: results ++ [result_type]}
  end

  def add_result(%__MODULE__{results: results} = changeset, {:parametric, _, _, _f} = result_type),
    do: %__MODULE__{changeset | results: results ++ [result_type]}

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
  def reorder_operands(%__MODULE__{operands: operands, name: name, attributes: attributes, context: ctx} = changeset) do
    if should_reorder?(operands) do
      {operands, sizes} =
        case Dump.lookup(name) do
          {:ok, dump} -> process_operands_with_dump(operands, dump)
          _ -> process_operands_with_keys(operands)
        end

      update_changeset_with_operands(changeset, operands, sizes, attributes, ctx)
    else
      changeset
    end
  end

  defp process_operands_with_keys(operands) do
    # Check for duplicate keys
    keys = Keyword.keys(operands)

    if length(keys) != length(Enum.uniq(keys)), do: raise("Duplicate keys found in operands: #{inspect(keys)}")

    # Extract all values in order
    grouped_values = Keyword.values(operands) |> Enum.map(&List.wrap/1)
    segment_sizes = Enum.map(grouped_values, &length/1)
    all_values = grouped_values |> List.flatten()

    {all_values, MLIR.ODS.segment_sizes(segment_sizes)}
  end

  defp update_changeset_with_operands(%__MODULE__{} = changeset, operands, segment_sizes, attributes, ctx) do
    operand_segment_sizes =
      attributes[:operand_segment_sizes] || attributes[:operandSegmentSizes]

    attributes =
      case operand_segment_sizes do
        :infer ->
          attributes
          |> Keyword.delete(:operand_segment_sizes)
          |> Keyword.delete(:operandSegmentSizes)
          |> Keyword.put(:operand_segment_sizes, Beaver.Deferred.resolve(segment_sizes, ctx))

        _ ->
          attributes
      end

    %__MODULE__{changeset | operands: operands, attributes: attributes}
  end

  defp should_reorder?([]), do: true

  defp should_reorder?(operands) do
    {tagged, untagged} = Enum.split_with(operands, &match?({_atom, _value}, &1))
    has_tagged = not Enum.empty?(tagged)
    has_untagged = not Enum.empty?(untagged)

    if has_tagged and has_untagged, do: raise(ArgumentError, "Cannot mix tagged and untagged operands")

    has_tagged
  end

  defp compare_tag(tag, operand_name) do
    tag = to_string(tag)
    operand_name == tag or Macro.camelize(operand_name) == Macro.camelize(tag)
  end

  defp process_operands_with_dump(operands, op_dump) do
    # 1. Normalize provided operands for efficient lookup (O(N))
    # This avoids the nested loop by ensuring keys are strings, matching op_dump.
    op_name = op_dump["name"]

    # 2. Process defined operands in a single pass (O(M log N))
    # A `for` comprehension clearly expresses the transformation.
    tagged_operands =
      for %{"name" => operand_name, "kind" => kind} <- op_dump["operands"] || [] do
        values = matching_operand_values(operands, operand_name)

        # Warn if a required single operand is missing
        if Enum.empty?(values) and kind == "Single",
          do: Logger.warning("Single operand '#{operand_name}' not set when creating operation #{op_name}")

        {String.to_atom(operand_name), values}
      end

    # 3. Assemble the final result from the processed list
    # This is cleaner as it operates on a well-structured intermediate variable.
    all_values = Keyword.values(tagged_operands)
    segment_sizes = MLIR.ODS.segment_sizes(Enum.map(all_values, &length/1))
    final_operands = List.flatten(all_values)

    {final_operands, segment_sizes}
  end

  defp matching_operand_values(operands, operand_name) do
    Enum.reduce(operands, [], fn {key, values}, acc ->
      if compare_tag(key, operand_name), do: acc ++ List.wrap(values), else: acc
    end)
  end
end
