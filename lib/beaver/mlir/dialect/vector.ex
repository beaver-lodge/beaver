defmodule Beaver.MLIR.Dialect.Vector do
  @moduledoc "Operations and transfer helpers for the MLIR Vector dialect."

  alias Beaver.MLIR

  use Beaver.MLIR.Dialect,
    dialect: "vector",
    ops: Beaver.MLIR.Dialect.Registry.ops("vector")

  @doc "Build the boolean array attribute used by vector transfer bounds."
  def in_bounds(bounds, opts \\ []) when is_list(bounds) do
    unless Enum.all?(bounds, &is_boolean/1) do
      raise ArgumentError, "vector in_bounds entries must be booleans"
    end

    Beaver.Deferred.from_opts(opts, fn ctx ->
      bounds
      |> Enum.map(&MLIR.Attribute.bool(&1, ctx: ctx))
      |> MLIR.Attribute.array(ctx: ctx)
    end)
  end

  @doc """
  Build a `vector.transfer_read` with named indices and transfer attributes.

      Vector.transfer_read_(source, padding,
        indices: [i], permutation_map: map, in_bounds: [true]
      ) >>> vector_type
  """
  def transfer_read_(%Beaver.SSA{arguments: [base, padding | options], ctx: ctx} = ssa) do
    options = Keyword.new(options)
    indices = Keyword.get(options, :indices, [])
    map = options |> Keyword.fetch!(:permutation_map) |> Beaver.Deferred.resolve(ctx)
    bounds = Keyword.get(options, :in_bounds, [])
    mask = Keyword.get(options, :mask)

    arguments = [
      {:base, base},
      {:indices, indices},
      {:padding, padding},
      {:mask, List.wrap(mask)},
      {:permutation_map, MLIR.Attribute.affine_map(map)},
      {:in_bounds, in_bounds(bounds, ctx: ctx)},
      {:operand_segment_sizes, :infer}
    ]

    MLIR.Operation.eval_ssa(%Beaver.SSA{ssa | op: transfer_read(), arguments: arguments})
  end
end
