defmodule Beaver.MLIR.Dialect.MemRef do
  use Beaver.MLIR.Dialect,
    dialect: "memref",
    ops: Beaver.MLIR.Dialect.Registry.ops("memref")
end
