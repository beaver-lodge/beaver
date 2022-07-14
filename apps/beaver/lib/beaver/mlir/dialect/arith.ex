defmodule Beaver.MLIR.Dialect.Arith do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect
  import MLIR.Sigils

  use Beaver.MLIR.Dialect.Generator,
    dialect: "arith",
    ops: Dialect.Registry.ops("arith", query: true) |> Enum.reject(fn x -> x in ~w{constant} end)

  # def constant(true) do
  #   MLIR.Operation.create("arith.constant", value: ~a{true}, result_types: ["i1"])
  #   |> MLIR.Operation.results()
  # end

  # def constant(false) do
  #   MLIR.Operation.create("arith.constant", value: ~a{false}, result_types: ["i1"])
  #   |> MLIR.Operation.results()
  # end

  # def constant(number) when is_number(number) do
  #   MLIR.Operation.create("arith.constant", value: ~a{#{number}}, result_types: ["i64"])
  #   |> MLIR.Operation.results()
  # end

  # def constant(arguments) when is_list(arguments) do
  #   MLIR.Operation.create("arith.constant", arguments)
  #   |> MLIR.Operation.results()
  # end

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
