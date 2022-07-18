defmodule Beaver.MLIR.Dialect.PDL do
  use Beaver.MLIR.Dialect.Generator,
    dialect: "pdl",
    ops: Beaver.MLIR.Dialect.Registry.ops("pdl", query: true)
end
