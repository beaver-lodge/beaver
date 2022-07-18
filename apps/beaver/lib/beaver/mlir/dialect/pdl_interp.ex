defmodule Beaver.MLIR.Dialect.PDLInterp do
  use Beaver.MLIR.Dialect.Generator,
    dialect: "pdl_interp",
    ops: Beaver.MLIR.Dialect.Registry.ops("pdl_interp", query: true)
end
