defmodule Beaver.MLIR.Dialect.Elixir do
  use Beaver.MLIR.Dialect,
    dialect: "elixir",
    ops: ~w{add}
end
