alias Beaver.MLIR.{Attribute, CAPI, Location, StringRef}
alias Beaver.Native

defmodule Beaver.MLIR.Operation do
  @moduledoc "Creates, inspects and modifies MLIR operations."
  alias Beaver.MLIR
  alias __MODULE__.State
  alias Beaver.Changeset
  import CAPI
  @behaviour Access

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native

  def create(%Beaver.SSA{} = ssa) do
    filler = if is_function(ssa.filler, 0), do: [ssa.filler], else: []
    changeset = %Changeset{name: ssa.op, location: ssa.loc || Location.unknown(), context: ssa.ctx}

    Enum.reduce(ssa.arguments ++ filler, changeset, &Changeset.add_argument(&2, &1))
    |> then(fn changeset -> Enum.reduce(ssa.results, changeset, &Changeset.add_result(&2, &1)) end)
    |> State.create()
    |> create()
    |> tap(&Beaver.MLIR.InsertionPoint.insert_operation(ssa.ip, &1))
  end

  def create(%Changeset{} = c), do: c |> State.create() |> create

  def create(%State{} = state), do: beaverOperationCreate(state)

  def results(%__MODULE__{} = op), do: Beaver.Walker.results(op)

  def results({:deferred, {_func_name, _arguments}} = deferred), do: deferred

  def name(%__MODULE__{} = op), do: mlirOperationGetName(op) |> mlirIdentifierStr() |> MLIR.to_string()

  defdelegate location(op), to: CAPI, as: :mlirOperationGetLocation

  @doc "Sets an operation's location and returns the operation."
  @spec set_location(__MODULE__.t(), Location.source()) :: __MODULE__.t()
  def set_location(%__MODULE__{} = op, loc) do
    loc = Location.resolve(loc, MLIR.context(op))
    :ok = mlirOperationSetLocation(op, loc)
    op
  end

  defdelegate parent(op), to: CAPI, as: :mlirOperationGetParentOperation
  defdelegate destroy(op), to: CAPI, as: :mlirOperationDestroy
  defdelegate clone(op), to: CAPI, as: :mlirOperationClone

  @doc "Returns the block containing `op`, or `nil` if `op` is detached."
  @spec block(t()) :: MLIR.Block.t() | nil
  def block(%__MODULE__{} = op), do: op |> mlirOperationGetBlock() |> nil_if_null()

  @doc "Returns the parent operation of `op`, or `nil` if `op` is top-level."
  @spec parent_op(t()) :: t() | nil
  def parent_op(%__MODULE__{} = op), do: op |> mlirOperationGetParentOperation() |> nil_if_null()

  @doc "Returns the next operation in the same block, or `nil` at the end of the block."
  @spec next_in_block(t()) :: t() | nil
  def next_in_block(%__MODULE__{} = op), do: op |> mlirOperationGetNextInBlock() |> nil_if_null()

  @doc "Returns the top-most ancestor operation containing `op` (or `op` itself if top-level)."
  @spec root(t()) :: t()
  def root(%__MODULE__{} = op) do
    case parent_op(op) do
      nil -> op
      parent -> root(parent)
    end
  end

  @doc "Returns the nearest enclosing operation (including `op`) that has a `SymbolTable`, or `nil`."
  @spec enclosing_symbol_table(t()) :: t() | nil
  def enclosing_symbol_table(%__MODULE__{} = op) do
    sym_table = mlirSymbolTableCreate(op)

    if MLIR.null?(sym_table) do
      case parent_op(op) do
        nil -> nil
        parent -> enclosing_symbol_table(parent)
      end
    else
      mlirSymbolTableDestroy(sym_table)
      op
    end
  end

  @doc """
  Copies attributes from `src` to `dst`, returning `dst`.

  Options:
    * `:only` - list of attribute names (atoms or strings) to include.
    * `:except` - list of attribute names (atoms or strings) to exclude.
  """
  @spec copy_attributes(t(), t(), keyword()) :: t()
  def copy_attributes(%__MODULE__{} = src, %__MODULE__{} = dst, opts \\ []) when is_list(opts) do
    only_set = opts[:only] && MapSet.new(opts[:only], &Kernel.to_string/1)
    except_set = MapSet.new(Keyword.get(opts, :except, []), &Kernel.to_string/1)

    for {id, attr} <- Beaver.Walker.attributes(src) do
      key = MLIR.to_string(id)

      if (is_nil(only_set) or MapSet.member?(only_set, key)) and not MapSet.member?(except_set, key) do
        put_attribute(dst, key, attr)
      end
    end

    dst
  end

  @doc "Returns all operation properties as an attribute, or nil if there are none."
  def properties(%__MODULE__{} = op), do: op |> beaverOperationGetProperties() |> nil_if_null()

  @doc """
  Replaces properties from their attribute representation, preserving discardable
  attributes. Failed conversion leaves the original properties unchanged.
  """
  def put_properties(%__MODULE__{} = op, properties) do
    ctx = MLIR.context(op)
    properties = Beaver.Deferred.resolve(properties, ctx)
    {result, diagnostics} = beaverOperationSetPropertiesWithDiagnostics(ctx, op, properties)
    if MLIR.LogicalResult.success?(result), do: {:ok, op}, else: {:error, diagnostics}
  end

  defdelegate result(op, pos), to: CAPI, as: :mlirOperationGetResult

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
  Tests structural equivalence, including locations, discardable attributes,
  properties and commutative operand ordering unless ignored via options.
  `ignore_commutativity: true` makes operand comparison order-sensitive,
  following MLIR's `IgnoreCommutativity` flag.
  """
  @spec equivalent?(__MODULE__.t(), __MODULE__.t(), equivalence_options()) :: boolean()
  def equivalent?(%__MODULE__{} = lhs, %__MODULE__{} = rhs, opts \\ []),
    do: mlirOperationIsStructurallyEquivalent(lhs, rhs, encode_equivalence_options(opts)) |> Native.to_term()

  @doc """
  Computes MLIR's structural hash using the options of `equivalent?/3`.
  Equivalent operations have equal hashes; the converse is not guaranteed.
  MLIR hashes external operands by identity and omits results and regions:
  use this for bucketing before equivalence checks, not as a content digest.
  """
  @spec structural_hash(__MODULE__.t(), equivalence_options()) :: non_neg_integer()
  def structural_hash(%__MODULE__{} = op, opts \\ []),
    do: beaverOperationStructuralHashValue(op, encode_equivalence_options(opts)) |> Native.to_term()

  defp encode_equivalence_options(opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "operation equivalence options must be a keyword list")

    Enum.reduce(opts, 0, fn
      {key, enabled}, flags when is_map_key(@equivalence_flags, key) and is_boolean(enabled) ->
        if enabled, do: Bitwise.bor(flags, Map.fetch!(@equivalence_flags, key)), else: flags

      {key, enabled}, _flags when is_map_key(@equivalence_flags, key) ->
        raise ArgumentError, "expected a boolean for #{inspect(key)}, got: #{inspect(enabled)}"

      {key, _enabled}, _flags ->
        raise ArgumentError, "unsupported operation equivalence option: #{inspect(key)}"
    end)
  end

  def from_module(%MLIR.Module{} = module), do: mlirModuleGetOperation(module)

  def from_module(%__MODULE__{} = op), do: op

  defp normalize_results([result]), do: result
  defp normalize_results(results), do: results

  @doc """
  Evaluates SSA, returning the operation for zero results, the value for one,
  or a list for multiple results. `{:op, types}` additionally returns the
  operation, preserving a result list when `types` is a list.
  """
  def eval_ssa(%Beaver.SSA{results: types} = ssa) do
    ssa =
      case types do
        [{:op, types}] -> %Beaver.SSA{ssa | results: List.wrap(types)}
        _ -> ssa
      end

    op = create(ssa)
    results = op |> results() |> Enum.to_list()

    case {types, results} do
      {[{:op, types}], _} when is_list(types) -> {op, results}
      {[{:op, _}], _} -> {op, normalize_results(results)}
      {_, []} -> op
      {_, _} -> normalize_results(results)
    end
  end

  @impl Access
  def fetch(op, attribute) do
    case attribute(op, attribute) do
      nil -> :error
      %Attribute{} = value -> {:ok, value}
    end
  end

  @impl Access
  def get_and_update(op, attribute, function) do
    attr = attribute(op, attribute)

    case function.(attr) do
      {_current_value, new_value} -> put_attribute(op, attribute, new_value)
      :pop -> remove_attribute(op, attribute)
    end

    {attr, op}
  end

  @impl Access
  def pop(op, attribute) do
    {:ok, attr} = fetch(op, attribute)
    remove_attribute(op, attribute)
    {attr, op}
  end

  @doc "Returns whether the operation schema declares an inherent attribute name."
  @spec inherent_attribute?(t(), String.t() | atom()) :: boolean()
  def inherent_attribute?(%__MODULE__{} = op, name),
    do: beaverOperationHasInherentAttributeByName(op, StringRef.create(name)) |> Native.to_term()

  @doc "Returns an attached inherent attribute, or `nil` when it is absent."
  @spec inherent_attribute(t(), String.t() | atom()) :: Attribute.t() | nil
  def inherent_attribute(%__MODULE__{} = op, name),
    do: op |> beaverOperationGetInherentAttributeByName(StringRef.create(name)) |> nil_if_null()

  @doc "Sets an attribute declared by the operation schema."
  @spec put_inherent_attribute(t(), String.t() | atom(), term()) :: :ok
  def put_inherent_attribute(%__MODULE__{} = op, name, value) do
    unless inherent_attribute?(op, name),
      do: raise(ArgumentError, "#{inspect(name)} is not an inherent attribute of #{op |> name()}")

    mlirOperationSetInherentAttributeByName(
      op,
      StringRef.create(name),
      Beaver.Deferred.resolve(value, MLIR.context(op))
    )
  end

  @doc "Returns a discardable metadata attribute, or `nil` when it is absent."
  @spec discardable_attribute(t(), String.t() | atom()) :: Attribute.t() | nil
  def discardable_attribute(%__MODULE__{} = op, name),
    do: op |> mlirOperationGetDiscardableAttributeByName(StringRef.create(name)) |> nil_if_null()

  @doc "Sets discardable metadata, rejecting names owned by the operation schema."
  @spec put_discardable_attribute(t(), String.t() | atom(), term()) :: :ok
  def put_discardable_attribute(%__MODULE__{} = op, name, value) do
    if inherent_attribute?(op, name),
      do: raise(ArgumentError, "#{inspect(name)} is inherent to #{op |> name()} and cannot be discardable")

    mlirOperationSetDiscardableAttributeByName(
      op,
      StringRef.create(name),
      Beaver.Deferred.resolve(value, MLIR.context(op))
    )
  end

  @doc "Removes discardable metadata and reports whether it existed."
  @spec remove_discardable_attribute(t(), String.t() | atom()) :: boolean()
  def remove_discardable_attribute(%__MODULE__{} = op, name),
    do: mlirOperationRemoveDiscardableAttributeByName(op, StringRef.create(name)) |> Native.to_term()

  @doc "Reads an attribute through its schema-defined semantic path."
  @spec attribute(t(), String.t() | atom()) :: Attribute.t() | nil
  def attribute(%__MODULE__{} = op, name),
    do: if(inherent_attribute?(op, name), do: inherent_attribute(op, name), else: discardable_attribute(op, name))

  @doc "Sets an attribute through its schema-defined semantic path."
  @spec put_attribute(t(), String.t() | atom(), term()) :: :ok
  def put_attribute(%__MODULE__{} = op, name, value) do
    if inherent_attribute?(op, name),
      do: put_inherent_attribute(op, name, value),
      else: put_discardable_attribute(op, name, value)
  end

  @doc "Removes an attribute through its schema-defined semantic path."
  @spec remove_attribute(t(), String.t() | atom()) :: boolean() | :ok
  def remove_attribute(%__MODULE__{} = op, name) do
    if inherent_attribute?(op, name),
      do: mlirOperationSetInherentAttributeByName(op, StringRef.create(name), Attribute.null()),
      else: remove_discardable_attribute(op, name)
  end

  defp nil_if_null(attribute), do: if(MLIR.null?(attribute), do: nil, else: attribute)

  def with_symbol_table(%__MODULE__{} = op, fun),
    do: Beaver.Native.with_resource(mlirSymbolTableCreate(op), &mlirSymbolTableDestroy/1, fun)

  @doc "Returns an enumerable of all SSA operands consumed by `op`."
  defdelegate operands(op), to: Beaver.Walker

  @doc "Returns the types of all SSA results produced by `op`."
  @spec result_types(t()) :: [MLIR.Type.t()]
  def result_types(%__MODULE__{} = op),
    do: op |> results() |> Enum.map(&MLIR.Value.type/1)

  @doc "Returns the types of all SSA operands consumed by `op`."
  @spec operand_types(t()) :: [MLIR.Type.t()]
  def operand_types(%__MODULE__{} = op),
    do: op |> operands() |> Enum.map(&MLIR.Value.type/1)

  @doc "Returns whether the operation is a terminator."
  def terminator?(%__MODULE__{} = op), do: MLIR.Trait.has?(op, :terminator)

  def implements_interface?(%__MODULE__{} = op, interface_id),
    do: mlirOperationImplementsInterface(op, interface_id) |> Native.to_term()

  def infer_type?(%__MODULE__{} = op), do: implements_interface?(op, mlirInferTypeOpInterfaceTypeID())

  def infer_shaped?(%__MODULE__{} = op),
    do: implements_interface?(op, mlirInferShapedTypeOpInterfaceTypeID())
end
