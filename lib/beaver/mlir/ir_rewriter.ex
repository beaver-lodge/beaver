alias Beaver.MLIR.{CAPI, RewriterBase}

defmodule Beaver.MLIR.IRRewriter do
  @moduledoc """
  Owns standalone MLIR IR rewriters.

  Use `with_rewriter/2` for imperative transformations outside a rewrite
  pattern. The rewriter is always destroyed when the callback returns, raises,
  throws, or exits.
  """

  alias Beaver.MLIR

  @type owner() :: MLIR.Context.t() | MLIR.Operation.t()

  @doc "Creates a standalone rewriter for a context or before an operation."
  @spec create(owner()) :: RewriterBase.t()
  def create(%MLIR.Context{} = ctx), do: CAPI.mlirIRRewriterCreate(ctx)

  def create(%MLIR.Operation{} = op), do: CAPI.mlirIRRewriterCreateFromOp(op)

  @doc "Destroys a standalone rewriter created by this module."
  @spec destroy(RewriterBase.t()) :: :ok
  defdelegate destroy(rewriter), to: CAPI, as: :mlirIRRewriterDestroy

  @doc "Runs `fun` with a standalone rewriter and always destroys it afterward."
  @spec with_rewriter(owner(), (RewriterBase.t() -> result)) :: result when result: var
  def with_rewriter(owner, fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(create(owner), &destroy/1, fun)
end
