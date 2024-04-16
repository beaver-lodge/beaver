# Here are modules provide struct and functions to work with IR entities not eagerly created. Usually it is an attribute/type doesn't get created until there is a MLIR context from block/op state.

defmodule Beaver.Deferred do
  alias Beaver.MLIR

  @type opts :: [ctx: MLIR.Context.t()]
  @type type :: MLIR.Type.t() | (MLIR.Context.t() -> MLIR.Type.t())
  @type operation :: MLIR.Operation.t() | (MLIR.Context.t() -> MLIR.Operation.t())
  @type attribute :: MLIR.Attribute.t() | (MLIR.Context.t() -> MLIR.Attribute.t())
  def from_opts(opts, f) do
    if ctx = Keyword.get(opts, :ctx) do
      f.(ctx)
    else
      f
    end
  end

  def create({:parametric, _, _, f}, ctx) when is_function(f) and not is_nil(ctx) do
    f.(ctx)
  end

  def create({:parametric, _, _, entity}, _ctx) do
    entity
  end

  def create(f, ctx) when is_function(f) do
    f.(ctx)
  end

  def create(entity, _ctx) do
    entity
  end
end
