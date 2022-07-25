defmodule Beaver.MLIR.Dialect.Complex do
  use Beaver.MLIR.Dialect,
    dialect: "complex",
    ops: Beaver.MLIR.Dialect.Registry.ops("complex")
end
