defmodule Beaver.MLIR.Dialect.PDL do
  use Beaver.MLIR.Dialect,
    dialect: "pdl",
    ops: Beaver.MLIR.Dialect.Registry.ops("pdl")
end
