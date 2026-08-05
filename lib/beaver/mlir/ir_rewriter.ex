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
  @spec create(owner()) :: MLIR.RewriterBase.t()
  def create(%MLIR.Context{} = context), do: MLIR.CAPI.mlirIRRewriterCreate(context)

  def create(%MLIR.Operation{} = operation),
    do: MLIR.CAPI.mlirIRRewriterCreateFromOp(operation)

  @doc "Destroys a standalone rewriter created by this module."
  @spec destroy(MLIR.RewriterBase.t()) :: :ok
  defdelegate destroy(rewriter), to: MLIR.CAPI, as: :mlirIRRewriterDestroy

  @doc "Runs `fun` with a standalone rewriter and always destroys it afterward."
  @spec with_rewriter(owner(), (MLIR.RewriterBase.t() -> result)) :: result when result: var
  def with_rewriter(owner, fun) when is_function(fun, 1) do
    rewriter = create(owner)

    try do
      fun.(rewriter)
    after
      destroy(rewriter)
    end
  end
end
