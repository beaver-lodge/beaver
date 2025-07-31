defmodule Beaver.Changeset do
  @moduledoc false
  alias Beaver.MLIR

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

  @type type_argument() :: MLIR.Type.t() | (MLIR.Context.t() -> MLIR.Type.t())
  @type attribute_argument() :: MLIR.Attribute.t() | (MLIR.Context.t() -> MLIR.Attribute.t())
  @type value_argument() :: MLIR.Value.t() | (MLIR.Context.t() -> MLIR.Value.t())
  @type tagged_attribute :: {atom(), type_argument() | attribute_argument()}
  @type branching_argument() :: MLIR.Block.t() | {MLIR.Block.t(), [MLIR.Value.t()]}
  @type region_argument() :: MLIR.Region.t() | {:regions, (-> [MLIR.Region.t()])}
  @type result_argument() :: MLIR.Type.t() | {:result_types, [MLIR.Type.t()]}
  @type loc_argument() :: MLIR.Location.t() | {:loc, MLIR.Location.t()}

  @type argument() ::
          value_argument()
          | tagged_attribute()
          | [tagged_attribute()]
          | branching_argument()
          | result_argument()
          | loc_argument()
  @spec add_argument(t(), argument()) :: t()

  @doc """
  Add an ODS argument to changeset.
  """

  def add_argument(%__MODULE__{context: context} = changeset, {tag, f})
      when is_atom(tag) and is_function(f, 1) do
    add_argument(changeset, {tag, Beaver.Deferred.create(f, context)})
  end

  def add_argument(%__MODULE__{context: context} = changeset, f) when is_function(f, 1) do
    add_argument(changeset, Beaver.Deferred.create(f, context))
  end

  def add_argument(%__MODULE__{regions: regions} = changeset, %MLIR.Region{} = region) do
    %__MODULE__{changeset | regions: regions ++ [region]}
  end

  def add_argument(%__MODULE__{} = changeset, {:loc, %Beaver.MLIR.Location{} = location}) do
    %__MODULE__{changeset | location: location}
  end

  def add_argument(%__MODULE__{regions: regions} = changeset, {:regions, region_filler})
      when is_function(region_filler, 0) do
    %__MODULE__{changeset | regions: regions ++ region_filler.()}
  end

  def add_argument(_changeset, {:regions, _}) do
    raise "the function to create regions should have a arity of 0"
  end

  def add_argument(%__MODULE__{results: results} = changeset, {:result_types, result_types})
      when is_list(result_types) do
    %__MODULE__{changeset | results: results ++ result_types}
  end

  def add_argument(%__MODULE__{results: results} = changeset, %MLIR.Type{} = result_type) do
    %__MODULE__{changeset | results: results ++ [result_type]}
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
        %__MODULE__{attributes: attributes} = changeset,
        [{name, _attr} | _tail] = attrs
      )
      when is_atom(name) do
    %__MODULE__{changeset | attributes: attributes ++ attrs}
  end

  def add_argument(
        %__MODULE__{operands: operands} = changeset,
        %MLIR.Value{} = operand
      ) do
    %__MODULE__{changeset | operands: operands ++ [operand]}
  end

  def add_argument(%__MODULE__{}, operand) do
    raise ArgumentError, """
    Invalid argument.

    The following argument types are supported:
    - {:loc, %MLIR.Location{}} - Location argument
    - {:regions, (%MLIR.Context{} -> [%MLIR.Region{}])} - Region filler function
    - {:result_types, [%MLIR.Type{}]} - Result types list
    - %MLIR.Type{} - Single result type
    - {%MLIR.Block{}, [%MLIR.Value{}]} - Successor block with args
    - %MLIR.Block{} - Successor block
    - {atom(), (%MLIR.Attribute{} | (%MLIR.Context{} -> %MLIR.Attribute{}))} - Tagged attribute
    - {atom(), %MLIR.Type{}} - Tagged type
    - {atom(), binary()} - Tagged string attribute
    - [{atom(), any()}] - List of attributes
    - %MLIR.Value{} or (%MLIR.Context{} -> %MLIR.Value{}) - Operand value

    Received: #{inspect(operand)}
    """
  end
end
