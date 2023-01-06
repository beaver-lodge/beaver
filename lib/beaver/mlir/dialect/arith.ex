defmodule Beaver.MLIR.Dialect.Arith do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect
  import MLIR.Sigils

  use Beaver.MLIR.Dialect,
    dialect: "arith",
    ops: Dialect.Registry.ops("arith")

  @constant "arith.constant"
  def constant(%Beaver.DSL.SSA{arguments: [true], evaluator: evaluator} = ssa) do
    evaluator.(@constant, %{ssa | arguments: [value: ~a{true}]})
  end

  def constant(%Beaver.DSL.SSA{arguments: [false], evaluator: evaluator} = ssa) do
    evaluator.(@constant, %{ssa | arguments: [value: ~a{false}]})
  end

  def constant(%Beaver.DSL.SSA{evaluator: evaluator} = ssa) do
    evaluator.(@constant, ssa)
  end

  def cmp_f_predicate(type) do
    i =
      case type do
        false -> 0
        :oeq -> 1
        :ogt -> 2
        :oge -> 3
        :olt -> 4
        :ole -> 5
        :one -> 6
        :ord -> 7
        :ueq -> 8
        :ugt -> 9
        :uge -> 1
        :ult -> 1
        :ule -> 1
        :une -> 1
        :uno -> 1
        true -> 15
      end

    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end
end
