defmodule Beaver.MLIR.Dialect.Complex do
  use Beaver.MLIR.Dialect.Generator,
    dialect: "complex",
    ops: Beaver.MLIR.Dialect.Registry.ops("complex")
end
