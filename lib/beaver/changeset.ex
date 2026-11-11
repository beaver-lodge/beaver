alias Beaver.MLIR.{Block, Location, Region, Type, Value}

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

  def add_argument(%__MODULE__{context: nil, location: %Location{} = loc} = changeset, argument),
    do: add_argument(%__MODULE__{changeset | context: MLIR.context(loc)}, argument)

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

  def add_argument(%__MODULE__{context: ctx, attributes: attributes} = changeset, {name, %kind{} = attr})
      when is_atom(name) and kind in [MLIR.Attribute, Type] do
    attr = Deferred.resolve(attr, ctx)
    %__MODULE__{changeset | attributes: attributes ++ [{name, attr}]}
  end

  def add_argument(%__MODULE__{attributes: attributes} = changeset, {name, attr})
      when is_atom(name) and is_binary(attr), do: %__MODULE__{changeset | attributes: attributes ++ [{name, attr}]}

  def add_argument(%__MODULE__{} = changeset, {name, shorthand})
      when is_atom(name) and (shorthand == :unit or is_boolean(shorthand) or is_tuple(shorthand)) do
    add_argument(changeset, {name, resolve_shorthand_attribute!(shorthand)})
  end

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

  @shorthand_types %{
    i1: "i1",
    i8: "i8",
    i16: "i16",
    i32: "i32",
    i64: "i64",
    i128: "i128",
    index: "index",
    f16: "f16",
    bf16: "bf16",
    f32: "f32",
    f64: "f64",
    ptr: "!llvm.ptr",
    none: "none"
  }

  @type deferred_type() :: {:parametric, term(), term(), term()}
  @type type() :: Type.t() | String.t() | atom() | deferred_type() | [Type.t() | String.t() | atom() | deferred_type()]
  @type result() :: :infer | type() | Deferred.contextual(type())
  @spec add_result(t(), result()) :: t()

  def add_result(%__MODULE__{context: nil, location: %Location{} = loc} = changeset, argument),
    do: add_result(%__MODULE__{changeset | context: MLIR.context(loc)}, argument)

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

  def add_result(%__MODULE__{} = changeset, shorthand) when is_map_key(@shorthand_types, shorthand),
    do: add_result(changeset, Map.fetch!(@shorthand_types, shorthand))

  def add_result(%__MODULE__{context: ctx, results: results} = changeset, type_str) when is_binary(type_str) do
    resolved = if ctx, do: Type.get(type_str, ctx: ctx), else: type_str
    %__MODULE__{changeset | results: results ++ [resolved]}
  end

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
    - (list of) type, type string, or shorthand atom (:i1, :i8, :i16, :i32, :i64, :i128, :index, :f32, :f64, :ptr)
    - function that returns type
    - :infer
    """
  end

  defp resolve_shorthand_attribute!(:unit), do: MLIR.Attribute.unit()
  defp resolve_shorthand_attribute!(b) when is_boolean(b), do: MLIR.Attribute.bool(b)
  defp resolve_shorthand_attribute!({:bool, b}) when is_boolean(b), do: MLIR.Attribute.bool(b)
  defp resolve_shorthand_attribute!({:i1, b}) when is_boolean(b), do: MLIR.Attribute.bool(b)
  defp resolve_shorthand_attribute!({:i1, n}) when n in [0, 1], do: MLIR.Attribute.bool(n == 1)
  defp resolve_shorthand_attribute!({:i8, n}) when is_integer(n), do: MLIR.Attribute.integer(Type.i(8), n)
  defp resolve_shorthand_attribute!({:i16, n}) when is_integer(n), do: MLIR.Attribute.integer(Type.i(16), n)
  defp resolve_shorthand_attribute!({:i32, n}) when is_integer(n), do: MLIR.Attribute.integer(Type.i32(), n)
  defp resolve_shorthand_attribute!({:i64, n}) when is_integer(n), do: MLIR.Attribute.integer(Type.i64(), n)

  defp resolve_shorthand_attribute!({:int, width, n}) when is_integer(width) and width > 0 and is_integer(n),
    do: MLIR.Attribute.integer(Type.i(width), n)

  defp resolve_shorthand_attribute!({:index, n}) when is_integer(n), do: MLIR.Attribute.index(n)
  defp resolve_shorthand_attribute!({:f32, x}) when is_number(x), do: MLIR.Attribute.float(Type.f32(), x * 1.0)
  defp resolve_shorthand_attribute!({:f64, x}) when is_number(x), do: MLIR.Attribute.float(Type.f64(), x * 1.0)
  defp resolve_shorthand_attribute!({:string, s}) when is_binary(s), do: MLIR.Attribute.string(s)

  defp resolve_shorthand_attribute!({tag, s}) when tag in [:symbol, :flat_symbol] and (is_binary(s) or is_atom(s)),
    do: MLIR.Attribute.flat_symbol_ref(s |> to_string() |> String.trim_leading("@"))

  defp resolve_shorthand_attribute!({:type, %Type{} = t}), do: MLIR.Attribute.type(t)

  defp resolve_shorthand_attribute!({:type, s}) when is_binary(s),
    do: Deferred.defer(&MLIR.Attribute.type(Type.get(s, ctx: &1)))

  defp resolve_shorthand_attribute!({:dense_i32, list}) when is_list(list),
    do: MLIR.Attribute.dense_array(list, Beaver.Native.I32)

  defp resolve_shorthand_attribute!({:dense_i64, list}) when is_list(list),
    do: MLIR.Attribute.dense_array(list, Beaver.Native.I64)

  defp resolve_shorthand_attribute!({:parse, s}) when is_binary(s), do: s

  defp resolve_shorthand_attribute!(other),
    do: raise(ArgumentError, "unsupported attribute shorthand: #{inspect(other)}")

  @doc false
  def reorder_operands(%__MODULE__{operands: operands, name: name, attributes: attrs, context: ctx} = changeset) do
    if should_reorder?(operands) do
      groups =
        case MLIR.ODS.Dump.lookup(name) do
          {:ok, dump} -> operand_groups(operands, dump)
          _ -> operand_groups(operands)
        end

      attrs =
        if (attrs[:operand_segment_sizes] || attrs[:operandSegmentSizes]) == :infer do
          sizes = groups |> Enum.map(&length/1) |> MLIR.ODS.segment_sizes() |> Deferred.resolve(ctx)

          attrs
          |> Keyword.drop([:operand_segment_sizes, :operandSegmentSizes])
          |> Keyword.put(:operand_segment_sizes, sizes)
        else
          attrs
        end

      %{changeset | operands: List.flatten(groups), attributes: attrs}
    else
      changeset
    end
  end

  defp operand_groups(operands) do
    keys = Keyword.keys(operands)
    if length(keys) != length(Enum.uniq(keys)), do: raise("Duplicate keys found in operands: #{inspect(keys)}")
    Enum.map(Keyword.values(operands), &List.wrap/1)
  end

  defp operand_groups(operands, dump) do
    for %{"name" => name, "kind" => kind} <- dump["operands"] || [] do
      values =
        for {key, values} <- operands, compare_tag(key, name), value <- List.wrap(values), do: value

      if values == [] and kind == "Single",
        do: Logger.warning("Single operand '#{name}' not set when creating operation #{dump["name"]}")

      values
    end
  end

  defp should_reorder?([]), do: true

  defp should_reorder?([first | rest]) do
    tagged = match?({_, _}, first)

    unless Enum.all?(rest, &(match?({_, _}, &1) == tagged)),
      do: raise(ArgumentError, "Cannot mix tagged and untagged operands")

    tagged
  end

  defp compare_tag(tag, operand_name),
    do: operand_name == to_string(tag) or Macro.camelize(operand_name) == Macro.camelize(to_string(tag))
end
