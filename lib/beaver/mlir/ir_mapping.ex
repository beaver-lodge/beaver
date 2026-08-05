defmodule Beaver.MLIR.IRMapping do
  @moduledoc """
  An owned mapping between values, blocks, and operations in two pieces of IR.

  Mappings are mutable native resources. Destroy them explicitly with
  `destroy/1`, or use `with_mapping/1` for scoped work. Mapped entities are
  borrowed: a mapping must not be queried after either side of a mapping has
  been destroyed.

  `clone/2` creates a detached, caller-owned operation. Insert that operation
  into a block or destroy it. `clone/3` inserts through a rewriter and updates
  the mapping as part of the rewrite.

  A mapping and a scoped insertion point can be combined in a rewrite callback
  without leaking native state or the rewriter's temporary position:

      base = MLIR.PatternRewriter.as_base(rewriter)
      old_result = MLIR.Operation.result(operation, 0)

      MLIR.PatternRewriter.with_insertion_point(rewriter, {:before, operation}, fn ->
        MLIR.IRMapping.with_mapping(fn mapping ->
          replacement = build_replacement.(base)

          mapping
          |> MLIR.IRMapping.map(old_result, replacement)
          |> MLIR.IRMapping.lookup(old_result)
          |> then(&MLIR.RewriterBase.replace(base, old_result, &1))
        end)
      end)
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  @type entity() :: MLIR.Value.t() | MLIR.Block.t() | MLIR.Operation.t()
  @type rewriter() :: MLIR.RewriterBase.t() | MLIR.PatternRewriter.t()

  @doc "Creates an empty owned IR mapping."
  @spec create() :: t()
  def create, do: CAPI.mlirIRMappingCreate()

  @doc "Destroys an owned IR mapping."
  @spec destroy(t()) :: :ok
  defdelegate destroy(mapping), to: CAPI, as: :mlirIRMappingDestroy

  @doc "Runs `fun` with a new mapping and always destroys it."
  @spec with_mapping((t() -> result)) :: result when result: var
  def with_mapping(fun) when is_function(fun, 1) do
    mapping = create()

    try do
      fun.(mapping)
    after
      destroy(mapping)
    end
  end

  @doc "Adds or replaces a value, block, or operation mapping."
  @spec map(t(), entity(), entity()) :: t()
  def map(%__MODULE__{} = mapping, %MLIR.Value{} = from, %MLIR.Value{} = to) do
    :ok = CAPI.mlirIRMappingMapValue(mapping, from, to)
    mapping
  end

  def map(%__MODULE__{} = mapping, %MLIR.Block{} = from, %MLIR.Block{} = to) do
    :ok = CAPI.mlirIRMappingMapBlock(mapping, from, to)
    mapping
  end

  def map(%__MODULE__{} = mapping, %MLIR.Operation{} = from, %MLIR.Operation{} = to) do
    :ok = CAPI.mlirIRMappingMapOperation(mapping, from, to)
    mapping
  end

  @doc "Looks up a mapped entity, returning `nil` when no mapping exists."
  @spec lookup(t(), entity()) :: entity() | nil
  def lookup(%__MODULE__{} = mapping, %MLIR.Value{} = from) do
    mapping |> CAPI.mlirIRMappingLookupOrNullValue(from) |> nil_if_null()
  end

  def lookup(%__MODULE__{} = mapping, %MLIR.Block{} = from) do
    mapping |> CAPI.mlirIRMappingLookupOrNullBlock(from) |> nil_if_null()
  end

  def lookup(%__MODULE__{} = mapping, %MLIR.Operation{} = from) do
    mapping |> CAPI.mlirIRMappingLookupOrNullOperation(from) |> nil_if_null()
  end

  @doc "Looks up a mapped entity, returning the input entity when absent."
  @spec lookup_or_default(t(), entity()) :: entity()
  def lookup_or_default(%__MODULE__{} = mapping, %MLIR.Value{} = from) do
    CAPI.mlirIRMappingLookupOrDefaultValue(mapping, from)
  end

  def lookup_or_default(%__MODULE__{} = mapping, %MLIR.Block{} = from) do
    CAPI.mlirIRMappingLookupOrDefaultBlock(mapping, from)
  end

  def lookup_or_default(%__MODULE__{} = mapping, %MLIR.Operation{} = from) do
    CAPI.mlirIRMappingLookupOrDefaultOperation(mapping, from)
  end

  @doc "Returns whether a mapping exists for a value, block, or operation."
  @spec contains?(t(), entity()) :: boolean()
  def contains?(%__MODULE__{} = mapping, %MLIR.Value{} = entity) do
    CAPI.mlirIRMappingContainsValue(mapping, entity) |> Beaver.Native.to_term()
  end

  def contains?(%__MODULE__{} = mapping, %MLIR.Block{} = entity) do
    CAPI.mlirIRMappingContainsBlock(mapping, entity) |> Beaver.Native.to_term()
  end

  def contains?(%__MODULE__{} = mapping, %MLIR.Operation{} = entity) do
    CAPI.mlirIRMappingContainsOperation(mapping, entity) |> Beaver.Native.to_term()
  end

  @doc "Erases a value, block, or operation mapping and returns `mapping`."
  @spec erase(t(), entity()) :: t()
  def erase(%__MODULE__{} = mapping, %MLIR.Value{} = entity) do
    :ok = CAPI.mlirIRMappingEraseValue(mapping, entity)
    mapping
  end

  def erase(%__MODULE__{} = mapping, %MLIR.Block{} = entity) do
    :ok = CAPI.mlirIRMappingEraseBlock(mapping, entity)
    mapping
  end

  def erase(%__MODULE__{} = mapping, %MLIR.Operation{} = entity) do
    :ok = CAPI.mlirIRMappingEraseOperation(mapping, entity)
    mapping
  end

  @doc "Clears all mappings and returns `mapping`."
  @spec clear(t()) :: t()
  def clear(%__MODULE__{} = mapping) do
    :ok = CAPI.beaverIRMappingClear(mapping)
    mapping
  end

  @doc "Clones `operation` as a detached caller-owned operation and updates `mapping`."
  @spec clone(t(), MLIR.Operation.t()) :: MLIR.Operation.t()
  def clone(%__MODULE__{} = mapping, %MLIR.Operation{} = operation) do
    CAPI.mlirOperationCloneWithMapping(operation, mapping)
  end

  @doc "Clones and inserts `operation` through `rewriter`, updating `mapping`."
  @spec clone(t(), MLIR.Operation.t(), rewriter()) :: MLIR.Operation.t()
  def clone(
        %__MODULE__{} = mapping,
        %MLIR.Operation{} = operation,
        %MLIR.PatternRewriter{} = rewriter
      ) do
    clone(mapping, operation, MLIR.PatternRewriter.as_base(rewriter))
  end

  def clone(
        %__MODULE__{} = mapping,
        %MLIR.Operation{} = operation,
        %MLIR.RewriterBase{} = rewriter
      ) do
    CAPI.mlirRewriterBaseCloneWithMapping(rewriter, operation, mapping)
  end

  defp nil_if_null(entity) do
    if MLIR.null?(entity), do: nil, else: entity
  end
end
