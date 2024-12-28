defmodule Beaver.Deferred do
  @moduledoc """
  Functions to work with IR entities not eagerly created. Usually it is an attribute/type doesn't get created until there is a MLIR context from block/op state.
  """
  alias Beaver.MLIR

  @type opts :: [ctx: MLIR.Context.t()]
  @type type :: MLIR.Type.t() | (MLIR.Context.t() -> MLIR.Type.t())
  @type operation :: MLIR.Operation.t() | (MLIR.Context.t() -> MLIR.Operation.t())
  @type attribute :: MLIR.Attribute.t() | (MLIR.Context.t() -> MLIR.Attribute.t())
  def from_opts(opts, f) do
    if ctx = fetch_context(opts) do
      f.(ctx)
    else
      f
    end
  end

  @spec fetch_context(opts :: opts) :: MLIR.Context.t() | Macro.t() | nil
  def fetch_context(opts) do
    opts[:ctx]
  end

  @spec fetch_block(opts :: opts) :: MLIR.Block.t() | Macro.t() | nil
  def fetch_block(opts) do
    if opts[:block] do
      raise ArgumentError, "use :blk instead of :block as key"
    end

    opts[:blk]
  end

  defp unwrap_f(f, ctx) do
    case f.(ctx) do
      {:ok, value} -> value
      {:error, reason} -> raise ArgumentError, reason
      entity -> entity
    end
  end

  def create({:parametric, _, _, f}, ctx) when is_function(f) and not is_nil(ctx) do
    unwrap_f(f, ctx)
  end

  def create({:parametric, _, _, entity}, _ctx) do
    entity
  end

  def create(f, ctx) when is_function(f) do
    unwrap_f(f, ctx)
  end

  def create(entity, _ctx) do
    entity
  end
end
