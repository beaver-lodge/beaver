defmodule Beaver.MLIR.Dialect.SCF do
  @moduledoc """
  Operations and semantic builders for the MLIR `scf` dialect.

  The generated operation functions remain available. `if_/2` and `for_/5`
  add a small Elixir-shaped layer that owns the mechanical region, block
  argument, result, and `scf.yield` construction.
  """

  use Beaver.MLIR.Dialect,
    dialect: "scf",
    ops: Beaver.MLIR.Dialect.Registry.ops("scf")

  @doc """
  Build an `scf.if` with implicit regions and yields.

  The value of each branch is yielded. Use `result_types: []` (the default)
  for a statement-like if and end each branch with an expression returning
  `nil` or `[]`.

      require Beaver.MLIR.Dialect.SCF

      SCF.if_(condition, result_types: [Type.i32()]) do
        lhs
      else
        rhs
      end

  An else branch is required when the operation has results.
  """
  defmacro if_(condition, branches) do
    build_if(condition, [], branches)
  end

  defmacro if_(condition, opts, branches) do
    build_if(condition, opts, branches)
  end

  defp build_if(condition, opts, branches) do
    then_body = Keyword.fetch!(branches, :do)
    else_body = Keyword.get(branches, :else)
    result_types = Keyword.get(opts, :result_types, [])

    if else_body == nil and result_types != [] do
      raise ArgumentError, "SCF.if_ requires an else branch when result_types is not empty"
    end

    regions =
      [branch_region(then_body)] ++
        if else_body == nil, do: [], else: [branch_region(else_body)]

    quote do
      Beaver.mlir do
        Beaver.MLIR.Dialect.SCF.if unquote(condition) do
          (unquote_splicing(regions))
        end >>> unquote(result_types)
      end
    end
  end

  @doc """
  Build an `scf.for` with implicit block arguments and yield.

  The body is a two-argument function receiving the induction variable and a
  list of loop-carried values. Its return value is passed to `scf.yield`.

      require Beaver.MLIR.Dialect.SCF

      SCF.for_(lower, upper, step, iter_args: [initial]) do
        fn iv, [acc] -> Arith.addi(acc, iv) >>> Type.index() end
      end

  Result types default to the types of `iter_args`. Pass `result_types:` to
  override them explicitly.
  """
  defmacro for_(lower, upper, step, opts, do: body) do
    iter_args = Keyword.get(opts, :iter_args, [])

    unless is_list(iter_args) do
      raise ArgumentError, "SCF.for_/5 expects :iter_args to be a list"
    end

    iv = Macro.var(:scf_iv, nil)

    carried =
      for index <- 0..(length(iter_args) - 1)//1 do
        Macro.var(:"scf_iter_#{index}", nil)
      end

    block_args =
      [quote(do: unquote(iv) >>> Beaver.MLIR.Type.index())] ++
        Enum.zip_with(carried, iter_args, fn variable, initial ->
          quote do
            unquote(variable) >>> Beaver.MLIR.Value.type(unquote(initial))
          end
        end)

    block_call = {:_scf_for_body, [], block_args}
    operands = [lower, upper, step] ++ iter_args

    result_types =
      Keyword.get_lazy(opts, :result_types, fn ->
        Enum.map(iter_args, fn initial ->
          quote(do: Beaver.MLIR.Value.type(unquote(initial)))
        end)
      end)

    quote do
      Beaver.mlir do
        Beaver.MLIR.Dialect.SCF.for unquote(operands) do
          region do
            block unquote(block_call) do
              Beaver.MLIR.Dialect.SCF.yield(unquote(body).(unquote(iv), unquote(carried))) >>> []
            end
          end
        end >>> unquote(result_types)
      end
    end
  end

  defp branch_region(body) do
    quote do
      region do
        block do
          Beaver.MLIR.Dialect.SCF.yield(unquote(body)) >>> []
        end
      end
    end
  end
end
