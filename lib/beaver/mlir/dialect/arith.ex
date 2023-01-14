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

  defp type_to_magic_num(false), do: 0
  defp type_to_magic_num(:oeq), do: 1
  defp type_to_magic_num(:ogt), do: 2
  defp type_to_magic_num(:oge), do: 3
  defp type_to_magic_num(:olt), do: 4
  defp type_to_magic_num(:ole), do: 5
  defp type_to_magic_num(:one), do: 6
  defp type_to_magic_num(:ord), do: 7
  defp type_to_magic_num(:ueq), do: 8
  defp type_to_magic_num(:ugt), do: 9
  defp type_to_magic_num(:uge), do: 10
  defp type_to_magic_num(:ult), do: 11
  defp type_to_magic_num(:ule), do: 12
  defp type_to_magic_num(:une), do: 13
  defp type_to_magic_num(:uno), do: 14
  defp type_to_magic_num(true), do: 15

  def cmp_f_predicate(type) do
    i = type_to_magic_num(type)
    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end
end
