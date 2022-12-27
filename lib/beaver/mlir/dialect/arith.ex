defmodule Beaver.MLIR.Dialect.Arith do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect
  import MLIR.Sigils

  use Beaver.MLIR.Dialect,
    dialect: "arith",
    ops: Dialect.Registry.ops("arith") |> Enum.reject(fn x -> x in ~w{constant} end)

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
      %{
        false => 0,
        :oeq => 1,
        :ogt => 2,
        :oge => 3,
        :olt => 4,
        :ole => 5,
        :one => 6,
        :ord => 7,
        :ueq => 8,
        :ugt => 9,
        :uge => 10,
        :ult => 11,
        :ule => 12,
        :une => 13,
        :uno => 14,
        true => 15
      }[type]

    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end
end

require Beaver.MLIR.Dialect

Beaver.MLIR.Dialect.define_op_modules(
  Beaver.MLIR.Dialect.Arith,
  "arith",
  Beaver.MLIR.Dialect.Registry.ops("arith")
)
