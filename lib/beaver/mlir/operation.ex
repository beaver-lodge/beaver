defmodule Beaver.MLIR.Operation do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  alias __MODULE__.State
  alias Beaver.Changeset
  import Beaver.MLIR.CAPI
  @behaviour Access

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  def create(%Beaver.SSA{
        op: op_name,
        ip: ip,
        arguments: arguments,
        results: results,
        filler: filler,
        ctx: ctx,
        loc: loc
      }) do
    filler =
      if is_function(filler, 0) do
        List.wrap(filler)
      else
        []
      end

    location = loc || MLIR.Location.unknown()
    changeset = %Changeset{name: op_name, location: location, context: ctx}

    Enum.reduce(arguments ++ filler, changeset, &Changeset.add_argument(&2, &1))
    |> then(fn changeset -> Enum.reduce(results, changeset, &Changeset.add_result(&2, &1)) end)
    |> State.create()
    |> create()
    |> tap(&Beaver.MLIR.InsertionPoint.insert_operation(ip, &1))
  end

  def create(%Changeset{} = c) do
    c |> State.create() |> create
  end

  def create(%State{} = state) do
    beaverOperationCreate(state)
  end

  def results(%__MODULE__{} = op) do
    Beaver.Walker.results(op)
  end

  def results({:deferred, {_func_name, _arguments}} = deferred) do
    deferred
  end

  def name(%__MODULE__{} = operation) do
    mlirOperationGetName(operation)
    |> mlirIdentifierStr()
    |> MLIR.to_string()
  end

  defdelegate location(op), to: MLIR.CAPI, as: :mlirOperationGetLocation

  @doc "Sets an operation's location and returns the operation."
  @spec set_location(__MODULE__.t(), MLIR.Location.source()) :: __MODULE__.t()
  def set_location(%__MODULE__{} = operation, location) do
    location = MLIR.Location.resolve(location, MLIR.context(operation))
    :ok = mlirOperationSetLocation(operation, location)
    operation
  end

  defdelegate parent(op), to: MLIR.CAPI, as: :mlirOperationGetParentOperation
  defdelegate destroy(op), to: MLIR.CAPI, as: :mlirOperationDestroy
  defdelegate clone(op), to: MLIR.CAPI, as: :mlirOperationClone
  defdelegate result(op, pos), to: MLIR.CAPI, as: :mlirOperationGetResult

  @typedoc "Options shared by structural equivalence and structural hashing."
  @type equivalence_option() ::
          {:ignore_locations, boolean()}
          | {:ignore_discardable_attributes, boolean()}
          | {:ignore_properties, boolean()}
          | {:ignore_commutativity, boolean()}

  @type equivalence_options() :: [equivalence_option()]

  @equivalence_flags %{
    ignore_locations: 1,
    ignore_discardable_attributes: 2,
    ignore_properties: 4,
    ignore_commutativity: 8
  }

  @doc """
  Tests two operations for structural equivalence without printing or parsing.

  The default comparison includes locations, discardable attributes,
  properties, and commutative operand ordering. Set an option to `true` to
  ignore that part of the comparison. In particular,
  `ignore_commutativity: true` makes operand comparison order-sensitive; this
  mirrors MLIR's `IgnoreCommutativity` flag.
  """
  @spec equivalent?(__MODULE__.t(), __MODULE__.t(), equivalence_options()) :: boolean()
  def equivalent?(%__MODULE__{} = lhs, %__MODULE__{} = rhs, opts \\ []) do
    mlirOperationIsStructurallyEquivalent(lhs, rhs, encode_equivalence_options(opts))
    |> Beaver.Native.to_term()
  end

  @doc """
  Computes MLIR's structural hash for an operation.

  Use the same options as `equivalent?/3`: operations equivalent under the
  same options have equal hashes. The converse is not guaranteed. MLIR hashes
  external operands by identity, omits results, and does not include regions,
  so this is intended for bucketing before an equivalence check rather than as
  a complete content digest.
  """
  @spec structural_hash(__MODULE__.t(), equivalence_options()) :: non_neg_integer()
  def structural_hash(%__MODULE__{} = operation, opts \\ []) do
    beaverOperationStructuralHashValue(operation, encode_equivalence_options(opts))
    |> Beaver.Native.to_term()
  end

  defp encode_equivalence_options(opts) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "operation equivalence options must be a keyword list"
    end

    Enum.reduce(opts, 0, fn
      {key, enabled}, flags when is_map_key(@equivalence_flags, key) and is_boolean(enabled) ->
        if enabled, do: Bitwise.bor(flags, Map.fetch!(@equivalence_flags, key)), else: flags

      {key, enabled}, _flags when is_map_key(@equivalence_flags, key) ->
        raise ArgumentError, "expected a boolean for #{inspect(key)}, got: #{inspect(enabled)}"

      {key, _enabled}, _flags ->
        raise ArgumentError, "unsupported operation equivalence option: #{inspect(key)}"
    end)
  end

  def from_module(%MLIR.Module{} = module) do
    mlirModuleGetOperation(module)
  end

  def from_module(%__MODULE__{} = op) do
    op
  end

  defp normalize_results(results) do
    case Enum.count(results) do
      0 ->
        []

      1 ->
        Enum.at(results, 0)

      n when n > 1 ->
        Enum.to_list(results)
    end
  end

  @doc """
  Evaluate the SSA and return the operation or its results based on the defined result types.

  Normalize the results of the given operation in the following way:
  - If the operation has no result, return the operation itself.
  - If the operation has one result, return that result.
  - If the operation has multiple results, return a list of results.
  """
  def eval_ssa(%Beaver.SSA{results: result_types} = ssa) do
    ssa =
      case result_types do
        [{:op, result_types}] ->
          %Beaver.SSA{ssa | results: List.wrap(result_types)}

        _ ->
          ssa
      end

    op = create(ssa)
    results = op |> results()

    case result_types do
      [{:op, result_types}] when is_list(result_types) ->
        {op, Enum.to_list(results)}

      [{:op, _}] ->
        {op, normalize_results(results)}

      _ ->
        case Enum.count(results) do
          0 ->
            op

          n when n > 0 ->
            normalize_results(results)
        end
    end
  end

  @impl Access
  def fetch(operation, attribute) do
    case attribute(operation, attribute) do
      nil -> :error
      %MLIR.Attribute{} = value -> {:ok, value}
    end
  end

  @impl Access
  def get_and_update(operation, attribute, function) do
    attr =
      case fetch(operation, attribute) do
        {:ok, attr} -> attr
        :error -> nil
      end

    case function.(attr) do
      {_current_value, new_value} ->
        put_attribute(operation, attribute, new_value)

      :pop ->
        remove_attribute(operation, attribute)
    end

    {attr, operation}
  end

  @impl Access
  def pop(operation, attribute) do
    {:ok, attr} = fetch(operation, attribute)
    remove_attribute(operation, attribute)
    {attr, operation}
  end

  @doc "Returns whether the operation schema declares an inherent attribute name."
  @spec inherent_attribute?(t(), String.t() | atom()) :: boolean()
  def inherent_attribute?(%__MODULE__{} = operation, name) do
    beaverOperationHasInherentAttributeByName(operation, MLIR.StringRef.create(name))
    |> Beaver.Native.to_term()
  end

  @doc "Returns an attached inherent attribute, or `nil` when it is absent."
  @spec inherent_attribute(t(), String.t() | atom()) :: MLIR.Attribute.t() | nil
  def inherent_attribute(%__MODULE__{} = operation, name) do
    operation
    |> beaverOperationGetInherentAttributeByName(MLIR.StringRef.create(name))
    |> nil_if_null()
  end

  @doc "Sets an attribute declared by the operation schema."
  @spec put_inherent_attribute(t(), String.t() | atom(), term()) :: :ok
  def put_inherent_attribute(%__MODULE__{} = operation, name, value) do
    unless inherent_attribute?(operation, name) do
      raise ArgumentError,
            "#{inspect(name)} is not an inherent attribute of #{operation |> name()}"
    end

    mlirOperationSetInherentAttributeByName(
      operation,
      MLIR.StringRef.create(name),
      Beaver.Deferred.resolve(value, MLIR.context(operation))
    )
  end

  @doc "Returns a discardable metadata attribute, or `nil` when it is absent."
  @spec discardable_attribute(t(), String.t() | atom()) :: MLIR.Attribute.t() | nil
  def discardable_attribute(%__MODULE__{} = operation, name) do
    operation
    |> mlirOperationGetDiscardableAttributeByName(MLIR.StringRef.create(name))
    |> nil_if_null()
  end

  @doc "Sets discardable metadata, rejecting names owned by the operation schema."
  @spec put_discardable_attribute(t(), String.t() | atom(), term()) :: :ok
  def put_discardable_attribute(%__MODULE__{} = operation, name, value) do
    if inherent_attribute?(operation, name) do
      raise ArgumentError,
            "#{inspect(name)} is inherent to #{operation |> name()} and cannot be discardable"
    end

    mlirOperationSetDiscardableAttributeByName(
      operation,
      MLIR.StringRef.create(name),
      Beaver.Deferred.resolve(value, MLIR.context(operation))
    )
  end

  @doc "Removes discardable metadata and reports whether it existed."
  @spec remove_discardable_attribute(t(), String.t() | atom()) :: boolean()
  def remove_discardable_attribute(%__MODULE__{} = operation, name) do
    mlirOperationRemoveDiscardableAttributeByName(operation, MLIR.StringRef.create(name))
    |> Beaver.Native.to_term()
  end

  @doc "Reads an attribute through its schema-defined semantic path."
  @spec attribute(t(), String.t() | atom()) :: MLIR.Attribute.t() | nil
  def attribute(%__MODULE__{} = operation, name) do
    if inherent_attribute?(operation, name) do
      inherent_attribute(operation, name)
    else
      discardable_attribute(operation, name)
    end
  end

  @doc "Sets an attribute through its schema-defined semantic path."
  @spec put_attribute(t(), String.t() | atom(), term()) :: :ok
  def put_attribute(%__MODULE__{} = operation, name, value) do
    if inherent_attribute?(operation, name) do
      put_inherent_attribute(operation, name, value)
    else
      put_discardable_attribute(operation, name, value)
    end
  end

  @doc "Removes an attribute through its schema-defined semantic path."
  @spec remove_attribute(t(), String.t() | atom()) :: boolean() | :ok
  def remove_attribute(%__MODULE__{} = operation, name) do
    if inherent_attribute?(operation, name) do
      mlirOperationSetInherentAttributeByName(
        operation,
        MLIR.StringRef.create(name),
        MLIR.Attribute.null()
      )
    else
      remove_discardable_attribute(operation, name)
    end
  end

  defp nil_if_null(attribute) do
    if MLIR.null?(attribute), do: nil, else: attribute
  end

  def with_symbol_table(%__MODULE__{} = op, fun) do
    symbol_table = mlirSymbolTableCreate(op)

    try do
      fun.(symbol_table)
    after
      mlirSymbolTableDestroy(symbol_table)
    end
  end

  @doc """
  Check if the operation is a terminator.
  """
  def terminator?(%__MODULE__{} = op) do
    MLIR.Trait.has?(op, :terminator)
  end

  def implements_interface?(%__MODULE__{} = op, interface_id) do
    mlirOperationImplementsInterface(op, interface_id)
    |> Beaver.Native.to_term()
  end

  def infer_type?(%__MODULE__{} = op) do
    implements_interface?(op, mlirInferTypeOpInterfaceTypeID())
  end

  def infer_shaped?(%__MODULE__{} = op) do
    implements_interface?(op, mlirInferShapedTypeOpInterfaceTypeID())
  end
end
