defmodule Beaver.MLIR.Dialect.Tensor do
  alias Beaver.MLIR.{Attribute, Type}

  use Beaver.MLIR.Dialect,
    dialect: "tensor",
    ops: Beaver.MLIR.Dialect.Registry.ops("tensor")

  def reassociation(list) do
    for grouping <- list do
      grouping
      |> Enum.map(&Attribute.integer(Type.i64(), &1))
      |> Attribute.array()
    end
    |> Attribute.array()
  end
end
