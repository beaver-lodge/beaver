defmodule Beaver.MLIR.Dialect.Bufferization do
  use Beaver.MLIR.Dialect,
    dialect: "bufferization",
    ops: Beaver.MLIR.Dialect.Registry.ops("bufferization")
end
