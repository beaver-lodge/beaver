alias Beaver.MLIR.{Block, Operation, Value}
alias Beaver.Native

defmodule Beaver.MLIR.DominanceInfo do
  @moduledoc """
  High-level access to MLIR dominance analysis.

  A dominance info owns native analysis state. It must not outlive the IR or
  context from which it was created, and must be destroyed with `destroy/1`.
  Prefer `with_info/2` when the analysis does not need to escape a scope.

  Mutating the analyzed IR may make cached answers stale. Call `invalidate/1`
  after a mutation before issuing another query.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Native

  @type analyzed_ir() :: MLIR.Module.t() | Operation.t()

  @doc "Creates an owned dominance analysis for an operation or module."
  @spec create(analyzed_ir()) :: t()
  def create(ir), do: ir |> Operation.from_module() |> CAPI.mlirDominanceInfoCreate()

  @doc "Destroys an owned dominance analysis."
  @spec destroy(t()) :: :ok
  defdelegate destroy(info), to: CAPI, as: :mlirDominanceInfoDestroy

  @doc "Runs `fun` with a new dominance analysis and always destroys it."
  @spec with_info(analyzed_ir(), (t() -> result)) :: result when result: var
  def with_info(ir, fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(create(ir), &destroy/1, fun)

  @doc "Returns whether `a` dominates `b`."
  @spec dominates?(t(), Operation.t(), Operation.t()) :: boolean()
  @spec dominates?(t(), Block.t(), Block.t()) :: boolean()
  def dominates?(%__MODULE__{} = info, %Operation{} = a, %Operation{} = b),
    do: CAPI.mlirDominanceInfoDominatesOperation(info, a, b) |> Native.to_term()

  def dominates?(%__MODULE__{} = info, %Block{} = a, %Block{} = b),
    do: CAPI.mlirDominanceInfoDominatesBlock(info, a, b) |> Native.to_term()

  @doc "Returns whether `a` properly dominates `b` (excluding equality)."
  @spec properly_dominates?(t(), Operation.t(), Operation.t()) :: boolean()
  @spec properly_dominates?(t(), Block.t(), Block.t()) :: boolean()
  def properly_dominates?(%__MODULE__{} = info, %Operation{} = a, %Operation{} = b),
    do: CAPI.mlirDominanceInfoProperlyDominatesOperation(info, a, b) |> Native.to_term()

  def properly_dominates?(%__MODULE__{} = info, %Block{} = a, %Block{} = b),
    do: CAPI.mlirDominanceInfoProperlyDominatesBlock(info, a, b) |> Native.to_term()

  @doc "Returns whether a value dominates an operation."
  @spec value_dominates?(t(), Value.t(), Operation.t()) :: boolean()
  def value_dominates?(%__MODULE__{} = info, %Value{} = value, %Operation{} = op),
    do: CAPI.mlirDominanceInfoValueDominates(info, value, op) |> Native.to_term()

  @doc "Returns whether a value properly dominates an operation."
  @spec value_properly_dominates?(t(), Value.t(), Operation.t()) :: boolean()
  def value_properly_dominates?(%__MODULE__{} = info, %Value{} = value, %Operation{} = op),
    do: CAPI.mlirDominanceInfoValueProperlyDominates(info, value, op) |> Native.to_term()

  @doc "Returns the nearest common dominator, or `nil` when none exists."
  @spec nearest_common_dominator(t(), Block.t(), Block.t()) :: Block.t() | nil
  def nearest_common_dominator(%__MODULE__{} = info, %Block{} = a, %Block{} = b) do
    block = CAPI.mlirDominanceInfoFindNearestCommonDominator(info, a, b)
    if MLIR.null?(block), do: nil, else: block
  end

  @doc "Returns whether a block is reachable from the entry block of its region."
  @spec reachable_from_entry?(t(), Block.t()) :: boolean()
  def reachable_from_entry?(%__MODULE__{} = info, %Block{} = block),
    do: CAPI.mlirDominanceInfoIsReachableFromEntry(info, block) |> Native.to_term()

  @doc "Invalidates all cached dominance information and returns `info`."
  @spec invalidate(t()) :: t()
  def invalidate(%__MODULE__{} = info) do
    :ok = CAPI.mlirDominanceInfoInvalidate(info)
    info
  end
end
