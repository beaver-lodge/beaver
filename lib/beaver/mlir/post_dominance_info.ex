defmodule Beaver.MLIR.PostDominanceInfo do
  @moduledoc """
  High-level access to MLIR post-dominance analysis.

  A post-dominance info owns native analysis state. It must not outlive the IR
  or context from which it was created, and must be destroyed with `destroy/1`.
  Prefer `with_info/2` for scoped use. Call `invalidate/1` after mutating the
  analyzed IR.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  @type analyzed_ir() :: MLIR.Module.t() | MLIR.Operation.t()

  @doc "Creates an owned post-dominance analysis for an operation or module."
  @spec create(analyzed_ir()) :: t()
  def create(ir) do
    ir |> MLIR.Operation.from_module() |> CAPI.mlirPostDominanceInfoCreate()
  end

  @doc "Destroys an owned post-dominance analysis."
  @spec destroy(t()) :: :ok
  defdelegate destroy(info), to: CAPI, as: :mlirPostDominanceInfoDestroy

  @doc "Runs `fun` with a new post-dominance analysis and always destroys it."
  @spec with_info(analyzed_ir(), (t() -> result)) :: result when result: var
  def with_info(ir, fun) when is_function(fun, 1) do
    info = create(ir)

    try do
      fun.(info)
    after
      destroy(info)
    end
  end

  @doc "Returns whether `a` post-dominates `b`."
  @spec post_dominates?(t(), MLIR.Operation.t(), MLIR.Operation.t()) :: boolean()
  @spec post_dominates?(t(), MLIR.Block.t(), MLIR.Block.t()) :: boolean()
  def post_dominates?(%__MODULE__{} = info, %MLIR.Operation{} = a, %MLIR.Operation{} = b) do
    CAPI.mlirPostDominanceInfoPostDominatesOperation(info, a, b)
    |> Beaver.Native.to_term()
  end

  def post_dominates?(%__MODULE__{} = info, %MLIR.Block{} = a, %MLIR.Block{} = b) do
    CAPI.mlirPostDominanceInfoPostDominatesBlock(info, a, b) |> Beaver.Native.to_term()
  end

  @doc "Returns whether `a` properly post-dominates `b` (excluding equality)."
  @spec properly_post_dominates?(t(), MLIR.Operation.t(), MLIR.Operation.t()) :: boolean()
  @spec properly_post_dominates?(t(), MLIR.Block.t(), MLIR.Block.t()) :: boolean()
  def properly_post_dominates?(
        %__MODULE__{} = info,
        %MLIR.Operation{} = a,
        %MLIR.Operation{} = b
      ) do
    CAPI.mlirPostDominanceInfoProperlyPostDominatesOperation(info, a, b)
    |> Beaver.Native.to_term()
  end

  def properly_post_dominates?(
        %__MODULE__{} = info,
        %MLIR.Block{} = a,
        %MLIR.Block{} = b
      ) do
    CAPI.mlirPostDominanceInfoProperlyPostDominatesBlock(info, a, b)
    |> Beaver.Native.to_term()
  end

  @doc "Invalidates all cached post-dominance information and returns `info`."
  @spec invalidate(t()) :: t()
  def invalidate(%__MODULE__{} = info) do
    :ok = CAPI.mlirPostDominanceInfoInvalidate(info)
    info
  end
end
