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

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  @type analyzed_ir() :: MLIR.Module.t() | MLIR.Operation.t()

  @doc "Creates an owned dominance analysis for an operation or module."
  @spec create(analyzed_ir()) :: t()
  def create(ir) do
    ir |> MLIR.Operation.from_module() |> CAPI.mlirDominanceInfoCreate()
  end

  @doc "Destroys an owned dominance analysis."
  @spec destroy(t()) :: :ok
  defdelegate destroy(info), to: CAPI, as: :mlirDominanceInfoDestroy

  @doc "Runs `fun` with a new dominance analysis and always destroys it."
  @spec with_info(analyzed_ir(), (t() -> result)) :: result when result: var
  def with_info(ir, fun) when is_function(fun, 1) do
    info = create(ir)

    try do
      fun.(info)
    after
      destroy(info)
    end
  end

  @doc "Returns whether `a` dominates `b`."
  @spec dominates?(t(), MLIR.Operation.t(), MLIR.Operation.t()) :: boolean()
  @spec dominates?(t(), MLIR.Block.t(), MLIR.Block.t()) :: boolean()
  def dominates?(%__MODULE__{} = info, %MLIR.Operation{} = a, %MLIR.Operation{} = b) do
    CAPI.mlirDominanceInfoDominatesOperation(info, a, b) |> Beaver.Native.to_term()
  end

  def dominates?(%__MODULE__{} = info, %MLIR.Block{} = a, %MLIR.Block{} = b) do
    CAPI.mlirDominanceInfoDominatesBlock(info, a, b) |> Beaver.Native.to_term()
  end

  @doc "Returns whether `a` properly dominates `b` (excluding equality)."
  @spec properly_dominates?(t(), MLIR.Operation.t(), MLIR.Operation.t()) :: boolean()
  @spec properly_dominates?(t(), MLIR.Block.t(), MLIR.Block.t()) :: boolean()
  def properly_dominates?(
        %__MODULE__{} = info,
        %MLIR.Operation{} = a,
        %MLIR.Operation{} = b
      ) do
    CAPI.mlirDominanceInfoProperlyDominatesOperation(info, a, b)
    |> Beaver.Native.to_term()
  end

  def properly_dominates?(%__MODULE__{} = info, %MLIR.Block{} = a, %MLIR.Block{} = b) do
    CAPI.mlirDominanceInfoProperlyDominatesBlock(info, a, b) |> Beaver.Native.to_term()
  end

  @doc "Returns whether a value dominates an operation."
  @spec value_dominates?(t(), MLIR.Value.t(), MLIR.Operation.t()) :: boolean()
  def value_dominates?(%__MODULE__{} = info, %MLIR.Value{} = value, %MLIR.Operation{} = op) do
    CAPI.mlirDominanceInfoValueDominates(info, value, op) |> Beaver.Native.to_term()
  end

  @doc "Returns whether a value properly dominates an operation."
  @spec value_properly_dominates?(t(), MLIR.Value.t(), MLIR.Operation.t()) :: boolean()
  def value_properly_dominates?(
        %__MODULE__{} = info,
        %MLIR.Value{} = value,
        %MLIR.Operation{} = op
      ) do
    CAPI.mlirDominanceInfoValueProperlyDominates(info, value, op)
    |> Beaver.Native.to_term()
  end

  @doc "Returns the nearest common dominator, or `nil` when none exists."
  @spec nearest_common_dominator(t(), MLIR.Block.t(), MLIR.Block.t()) :: MLIR.Block.t() | nil
  def nearest_common_dominator(%__MODULE__{} = info, %MLIR.Block{} = a, %MLIR.Block{} = b) do
    block = CAPI.mlirDominanceInfoFindNearestCommonDominator(info, a, b)
    if MLIR.null?(block), do: nil, else: block
  end

  @doc "Returns whether a block is reachable from the entry block of its region."
  @spec reachable_from_entry?(t(), MLIR.Block.t()) :: boolean()
  def reachable_from_entry?(%__MODULE__{} = info, %MLIR.Block{} = block) do
    CAPI.mlirDominanceInfoIsReachableFromEntry(info, block) |> Beaver.Native.to_term()
  end

  @doc "Invalidates all cached dominance information and returns `info`."
  @spec invalidate(t()) :: t()
  def invalidate(%__MODULE__{} = info) do
    :ok = CAPI.mlirDominanceInfoInvalidate(info)
    info
  end
end
