defmodule Beaver.MLIR.Dialect.SCF do
  use Beaver.MLIR.Dialect.Generator,
    dialect: "scf",
    ops: Beaver.MLIR.Dialect.Registry.ops("scf")
end
