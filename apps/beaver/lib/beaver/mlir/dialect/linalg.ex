defmodule Beaver.MLIR.Dialect.Linalg do
  use Beaver.MLIR.Dialect.Generator,
    dialect: "linalg",
    ops: Beaver.MLIR.Dialect.Registry.ops("linalg", query: true)
end
