defmodule Beaver.MLIR.Dialect.Ex.MaterializeBoundVariables do
  @moduledoc """
  Erases `ex.var`/`ex.bind` pairs into SSA before dialect conversion.

  A binding `ex.bind(ex.var, value)` is replaced by `value` and both operations
  are removed. A bare `ex.var` without a consuming `ex.bind` is left untouched
  and reported as unsupported by `Beaver.MLIR.Conversion.Ex`.

  `run!/1` materializes every `ex.func` in a module or operation. Run it before
  `Beaver.MLIR.Conversion.Ex.plan/1`.
  """

  alias Beaver.MLIR
  alias Beaver.Walker

  @spec run!(MLIR.Module.t() | MLIR.Operation.t()) :: MLIR.Module.t() | MLIR.Operation.t()
  def run!(%MLIR.Module{} = module) do
    module |> ex_funcs() |> Enum.each(&materialize_func/1)
    module
  end

  def run!(%MLIR.Operation{} = operation) do
    if(MLIR.Operation.name(operation) == "ex.func",
      do: [operation],
      else: ex_funcs(operation)
    )
    |> Enum.each(&materialize_func/1)

    operation
  end

  defp ex_funcs(operation) do
    operation
    |> operations()
    |> Enum.filter(&(MLIR.Operation.name(&1) == "ex.func"))
  end

  defp materialize_func(ex_func) do
    ex_func
    |> operations()
    |> Enum.filter(&(MLIR.Operation.name(&1) == "ex.bind"))
    |> Enum.each(&materialize_bind(&1, ex_func))
  end

  defp operations(operation) do
    {_, operations} =
      Walker.postwalk(operation, [], fn
        %MLIR.Operation{} = op, acc -> {op, [op | acc]}
        element, acc -> {element, acc}
      end)

    Enum.reverse(operations)
  end

  defp materialize_bind(bind, ex_func) do
    [variable, value] = bind |> Walker.operands() |> Enum.to_list()
    variable_op = variable |> MLIR.CAPI.mlirOpResultGetOwner()

    unless MLIR.Operation.name(variable_op) == "ex.var" do
      raise ArgumentError, "ex.bind first operand must be produced by ex.var"
    end

    [bind_result] = bind |> Walker.results() |> Enum.to_list()

    MLIR.IRRewriter.with_rewriter(ex_func, fn rewriter ->
      MLIR.RewriterBase.replace_all_uses_with(rewriter, bind_result, value)
      MLIR.RewriterBase.erase_op(rewriter, bind)
      MLIR.RewriterBase.erase_op(rewriter, variable_op)
    end)

    :ok
  end
end
