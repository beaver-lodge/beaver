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
    cond do
      Enum.all?(argument, &match?({_atom, %MLIR.Value{}}, &1)) ->
        %__MODULE__{changeset | operands: changeset.operands ++ argument}

      Enum.all?(argument, &match?(%MLIR.Value{}, &1)) ->
        %__MODULE__{changeset | operands: changeset.operands ++ argument}

      Enum.all?(
        argument,
        &(match?({_atom, %MLIR.Attribute{}}, &1) or match?({_atom, %MLIR.Value{}}, &1))
      ) ->
        for a <- argument, reduce: changeset do
          changeset ->
            case a do
              {name, %MLIR.Attribute{} = attr} when is_atom(name) ->
                %__MODULE__{changeset | attributes: changeset.attributes ++ [{name, attr}]}

              {name, %MLIR.Value{} = value} when is_atom(name) ->
                %__MODULE__{changeset | operands: changeset.operands ++ [{name, value}]}

              _ ->
                raise ArgumentError, "Invalid argument in attribute list: #{inspect(a)}"
            end
        end

      Enum.all?(argument, &match?(%MLIR.Region{}, &1)) ->
        %__MODULE__{changeset | regions: changeset.regions ++ argument}

      Enum.all?(argument, &match?(%MLIR.Type{}, &1)) ->
        %__MODULE__{changeset | results: changeset.results ++ argument}

      true ->
        for arg <- argument, reduce: changeset do
          changeset -> add_argument(changeset, arg)
        end
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
    if Enum.all?(argument, &match?(%MLIR.Type{}, &1)) do
      %__MODULE__{changeset | results: changeset.results ++ argument}
    else
      for arg <- argument, reduce: changeset do
        changeset -> add_result(changeset, arg)
      end
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
  def reorder_operands(%__MODULE__{operands: operands, name: name} = changeset) do
    # TODO: only lookup if operands have both tagged and untagged
    has_tag_operand = Enum.any?(operands, &match?({_atom, %MLIR.Value{}}, &1))

    if has_tag_operand do
      case MLIR.ODS.Dump.lookup(name) do
        {:ok, op_dump} ->
          # TODO: validate there is no duplication in dump
          operands =
            for %{"name" => operand_name} <- op_dump["operands"] do
              # TODO: print operand not consumed
              Enum.filter(operands, fn
                {tag, %MLIR.Value{}} ->
                  tag = to_string(tag)
                  operand_name == tag or operand_name == Macro.camelize(tag)
              end)
            end
            |> List.flatten()
            |> Enum.map(fn {_, v} -> v end)

          %__MODULE__{changeset | operands: operands}

        _ ->
          changeset
      end
    else
      changeset
    end
  end
end
