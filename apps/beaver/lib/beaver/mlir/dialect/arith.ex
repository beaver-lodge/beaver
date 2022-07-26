defmodule Beaver.MLIR.Dialect.Arith do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect
  import MLIR.Sigils

  use Beaver.MLIR.Dialect,
    dialect: "arith",
    ops: Dialect.Registry.ops("arith") |> Enum.reject(fn x -> x in ~w{constant} end)

  def constant(%Beaver.DSL.SSA{arguments: [true]} = ssa) do
    MLIR.Operation.create("arith.constant", %{ssa | arguments: [value: ~a{true}]})
    |> MLIR.Operation.results()
  end

  def constant(%Beaver.DSL.SSA{arguments: [false]} = ssa) do
    MLIR.Operation.create("arith.constant", %{ssa | arguments: [value: ~a{false}]})
    |> MLIR.Operation.results()
  end

  def constant(%Beaver.DSL.SSA{} = ssa) do
    MLIR.Operation.create("arith.constant", ssa)
    |> MLIR.Operation.results()
  end
end
