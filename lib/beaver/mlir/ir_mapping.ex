alias Beaver.MLIR.{Block, Operation, Value}
alias Beaver.Native

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

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Native

  @type entity() :: Value.t() | Block.t() | Operation.t()
  @type rewriter() :: MLIR.RewriterBase.t() | MLIR.PatternRewriter.t()

  @doc "Creates an empty owned IR mapping."
  @spec create() :: t()
  def create, do: CAPI.mlirIRMappingCreate()

  @doc "Destroys an owned IR mapping."
  @spec destroy(t()) :: :ok
  defdelegate destroy(mapping), to: CAPI, as: :mlirIRMappingDestroy

  @doc "Runs `fun` with a new mapping and always destroys it."
  @spec with_mapping((t() -> result)) :: result when result: var
  def with_mapping(fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(create(), &destroy/1, fun)

  @entity_kinds [Value, Block, Operation]
  for kind <- @entity_kinds, verb <- ~w(Map LookupOrNull LookupOrDefault Contains Erase) do
    defp mapping_function(unquote(kind), unquote(verb)),
      do: unquote(:"mlirIRMapping#{verb}#{List.last(Module.split(kind))}")
  end

  @doc "Adds or replaces a value, block, or operation mapping."
  @spec map(t(), entity(), entity()) :: t()
  def map(%__MODULE__{} = mapping, %kind{} = from, %kind{} = to) when kind in @entity_kinds do
    :ok = apply(CAPI, mapping_function(kind, "Map"), [mapping, from, to])
    mapping
  end

  @doc "Looks up a mapped entity, returning `nil` when no mapping exists."
  @spec lookup(t(), entity()) :: entity() | nil
  def lookup(%__MODULE__{} = mapping, %kind{} = from) when kind in @entity_kinds,
    do: apply(CAPI, mapping_function(kind, "LookupOrNull"), [mapping, from]) |> nil_if_null()

  @doc "Looks up a mapped entity, returning the input entity when absent."
  @spec lookup_or_default(t(), entity()) :: entity()
  def lookup_or_default(%__MODULE__{} = mapping, %kind{} = from) when kind in @entity_kinds,
    do: apply(CAPI, mapping_function(kind, "LookupOrDefault"), [mapping, from])

  @doc "Returns whether a mapping exists for a value, block, or operation."
  @spec contains?(t(), entity()) :: boolean()
  def contains?(%__MODULE__{} = mapping, %kind{} = entity) when kind in @entity_kinds,
    do: apply(CAPI, mapping_function(kind, "Contains"), [mapping, entity]) |> Native.to_term()

  @doc "Erases a value, block, or operation mapping and returns `mapping`."
  @spec erase(t(), entity()) :: t()
  def erase(%__MODULE__{} = mapping, %kind{} = entity) when kind in @entity_kinds do
    :ok = apply(CAPI, mapping_function(kind, "Erase"), [mapping, entity])
    mapping
  end

  @doc "Clears all mappings and returns `mapping`."
  @spec clear(t()) :: t()
  def clear(%__MODULE__{} = mapping) do
    :ok = CAPI.beaverIRMappingClear(mapping)
    mapping
  end

  @doc "Clones `operation` as a detached caller-owned operation and updates `mapping`."
  @spec clone(t(), Operation.t()) :: Operation.t()
  def clone(%__MODULE__{} = mapping, %Operation{} = op),
    do: CAPI.mlirOperationCloneWithMapping(op, mapping)

  @doc "Clones and inserts `operation` through `rewriter`, updating `mapping`."
  @spec clone(t(), Operation.t(), rewriter()) :: Operation.t()
  def clone(%__MODULE__{} = mapping, %Operation{} = op, %MLIR.PatternRewriter{} = rewriter),
    do: clone(mapping, op, MLIR.PatternRewriter.as_base(rewriter))

  def clone(%__MODULE__{} = mapping, %Operation{} = op, %MLIR.RewriterBase{} = rewriter),
    do: CAPI.mlirRewriterBaseCloneWithMapping(rewriter, op, mapping)

  defp nil_if_null(entity), do: if(MLIR.null?(entity), do: nil, else: entity)
end
