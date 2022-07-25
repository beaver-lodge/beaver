defmodule Beaver.MLIR.Dialect.Linalg do
  use Beaver.MLIR.Dialect,
    dialect: "linalg",
    ops: Beaver.MLIR.Dialect.Registry.ops("linalg")
end
