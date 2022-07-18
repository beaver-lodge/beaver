defmodule Beaver.MLIR.Dialect.Affine do
  use Beaver.MLIR.Dialect.Generator,
    dialect: "affine",
    ops: Beaver.MLIR.Dialect.Registry.ops("affine", query: true)
end
