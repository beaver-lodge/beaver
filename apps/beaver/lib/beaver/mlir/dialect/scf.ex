defmodule Beaver.MLIR.Dialect.SCF do
  use Beaver.MLIR.Dialect,
    dialect: "scf",
    ops: Beaver.MLIR.Dialect.Registry.ops("scf")
end
