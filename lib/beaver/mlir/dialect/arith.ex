defmodule Beaver.MLIR.Dialect.Arith do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect
  import MLIR.Sigils

  use Beaver.MLIR.Dialect,
    dialect: "arith",
    ops: Dialect.Registry.ops("arith")

  @constant "arith.constant"
  def constant(%Beaver.SSA{arguments: [true], evaluator: evaluator} = ssa) do
    evaluator.(%Beaver.SSA{ssa | op: @constant, arguments: [value: ~a{true}]})
  end

  def constant(%Beaver.SSA{arguments: [false], evaluator: evaluator} = ssa) do
    evaluator.(%Beaver.SSA{ssa | op: @constant, arguments: [value: ~a{false}]})
  end

  def constant(%Beaver.SSA{evaluator: evaluator} = ssa) do
    evaluator.(%Beaver.SSA{ssa | op: @constant})
  end

  defp f_type_to_magic_num(false), do: 0
  defp f_type_to_magic_num(:oeq), do: 1
  defp f_type_to_magic_num(:ogt), do: 2
  defp f_type_to_magic_num(:oge), do: 3
  defp f_type_to_magic_num(:olt), do: 4
  defp f_type_to_magic_num(:ole), do: 5
  defp f_type_to_magic_num(:one), do: 6
  defp f_type_to_magic_num(:ord), do: 7
  defp f_type_to_magic_num(:ueq), do: 8
  defp f_type_to_magic_num(:ugt), do: 9
  defp f_type_to_magic_num(:uge), do: 10
  defp f_type_to_magic_num(:ult), do: 11
  defp f_type_to_magic_num(:ule), do: 12
  defp f_type_to_magic_num(:une), do: 13
  defp f_type_to_magic_num(:uno), do: 14
  defp f_type_to_magic_num(true), do: 15

  def cmp_f_predicate(type) do
    i = f_type_to_magic_num(type)
    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end

  defp i_type_to_magic_num(:eq), do: 0
  defp i_type_to_magic_num(:ne), do: 1
  defp i_type_to_magic_num(:slt), do: 2
  defp i_type_to_magic_num(:sle), do: 3
  defp i_type_to_magic_num(:sgt), do: 4
  defp i_type_to_magic_num(:sge), do: 5
  defp i_type_to_magic_num(:ult), do: 6
  defp i_type_to_magic_num(:ule), do: 7
  defp i_type_to_magic_num(:ugt), do: 8
  defp i_type_to_magic_num(:uge), do: 9

  def cmp_i_predicate(type) do
    i = i_type_to_magic_num(type)
    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end
end
