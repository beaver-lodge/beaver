defmodule Beaver.MLIR.Dialect.Affine do
  use Beaver.MLIR.Dialect,
    dialect: "affine",
    ops: Beaver.MLIR.Dialect.Registry.ops("affine")
end
