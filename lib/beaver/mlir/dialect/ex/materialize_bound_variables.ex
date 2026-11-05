alias Beaver.MLIR.{Operation, RewriterBase}

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

  @spec run!(MLIR.Module.t() | Operation.t()) :: MLIR.Module.t() | Operation.t()
  def run!(%MLIR.Module{} = module) do
    module |> ex_funcs() |> Enum.each(&materialize_func/1)
    module
  end

  def run!(%Operation{} = op) do
    if(Operation.name(op) == "ex.func", do: [op], else: ex_funcs(op))
    |> Enum.each(&materialize_func/1)

    op
  end

  defp ex_funcs(op), do: op |> operations() |> Enum.filter(&(Operation.name(&1) == "ex.func"))

  defp materialize_func(ex_func) do
    ex_func
    |> operations()
    |> Enum.filter(&(Operation.name(&1) == "ex.bind"))
    |> Enum.each(&materialize_bind(&1, ex_func))
  end

  defp operations(operation) do
    {_, operations} =
      Walker.postwalk(operation, [], fn
        %Operation{} = op, acc -> {op, [op | acc]}
        element, acc -> {element, acc}
      end)

    Enum.reverse(operations)
  end

  defp materialize_bind(bind, ex_func) do
    [variable, value] = bind |> Walker.operands() |> Enum.to_list()
    variable_op = variable |> MLIR.CAPI.mlirOpResultGetOwner()

    unless Operation.name(variable_op) == "ex.var",
      do: raise(ArgumentError, "ex.bind first operand must be produced by ex.var")

    [bind_result] = bind |> Walker.results() |> Enum.to_list()

    MLIR.IRRewriter.with_rewriter(ex_func, fn rewriter ->
      RewriterBase.replace_all_uses_with(rewriter, bind_result, value)
      RewriterBase.erase_op(rewriter, bind)
      RewriterBase.erase_op(rewriter, variable_op)
    end)

    :ok
  end
end
