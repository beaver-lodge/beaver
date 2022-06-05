defmodule Beaver.MLIR.Dialect.Arith do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  import MLIR.Sigils

  def constant(true) do
    MLIR.Operation.create("arith.constant", value: ~a{true}, result_types: ["i1"])
  end

  def constant(false) do
    MLIR.Operation.create("arith.constant", value: ~a{false}, result_types: ["i1"])
  end

  def constant(number) when is_number(number) do
    MLIR.Operation.create("arith.constant", value: ~a{#{number}}, result_types: ["i64"])
  end

  def constant(arguments) when is_list(arguments) do
    MLIR.Operation.create("arith.constant", arguments)
  end

  def addi(arguments) do
    MLIR.Operation.create("arith.addi", arguments)
  end

  def addi(lhs, rhs, opts \\ []) do
  end
end
