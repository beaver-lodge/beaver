defmodule Beaver.MLIR.Dialect.Arith do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect
  import Beaver.Sigils

  use Dialect, dialect: "arith", ops: Dialect.Registry.ops("arith")

  @constant "arith.constant"
  def constant(%Beaver.SSA{arguments: [true], evaluator: evaluator} = ssa),
    do: evaluator.(%Beaver.SSA{ssa | op: @constant, arguments: [value: ~a{true}]})

  def constant(%Beaver.SSA{arguments: [false], evaluator: evaluator} = ssa),
    do: evaluator.(%Beaver.SSA{ssa | op: @constant, arguments: [value: ~a{false}]})

  def constant(%Beaver.SSA{evaluator: evaluator} = ssa), do: evaluator.(%Beaver.SSA{ssa | op: @constant})

  for {type, code} <-
        Enum.with_index([false] ++ ~w(oeq ogt oge olt ole one ord ueq ugt uge ult ule une uno)a ++ [true]) do
    defp f_type_to_magic_num(unquote(type)), do: unquote(code)
  end

  def cmp_f_predicate(type) do
    i = f_type_to_magic_num(type)
    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end

  for {type, code} <- Enum.with_index([:eq, :ne, :slt, :sle, :sgt, :sge, :ult, :ule, :ugt, :uge]) do
    defp i_type_to_magic_num(unquote(type)), do: unquote(code)
  end

  def cmp_i_predicate(type) do
    i = i_type_to_magic_num(type)
    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end

  @compare_operators [:!=, :==, :>, :>=, :<, :<=]
  defp operator_to_type(:!=, _), do: :ne
  defp operator_to_type(:==, _), do: :eq
  defp operator_to_type(:>, signed), do: :"#{if signed, do: :s, else: :u}gt"
  defp operator_to_type(:>=, signed), do: :"#{if signed, do: :s, else: :u}ge"
  defp operator_to_type(:<, signed), do: :"#{if signed, do: :s, else: :u}lt"
  defp operator_to_type(:<=, signed), do: :"#{if signed, do: :s, else: :u}le"

  def operator_to_predicate(operator, signed \\ true)

  def operator_to_predicate(operator, signed) when operator in @compare_operators,
    do: operator_to_type(operator, signed) |> cmp_i_predicate()
end
