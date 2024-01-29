defmodule Beaver.MLIR.Dialect do
  @moduledoc """
  This module defines macro to generate code for an MLIR dialect.
  You might override `eval_ssa/1` function to introduce your custom op generation
  """

  require Logger

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  @callback eval_ssa(Beaver.SSA.t()) :: any()
  defmacro __using__(opts) do
    dialect = Keyword.fetch!(opts, :dialect)
    ops = Keyword.fetch!(opts, :ops)

    quote(bind_quoted: [dialect: dialect, ops: ops]) do
      @behaviour Beaver.MLIR.Dialect

      def eval_ssa(%Beaver.SSA{evaluator: evaluator} = ssa) do
        evaluator.(ssa)
      end

      defoverridable eval_ssa: 1
      require Logger

      dialect_module_name = dialect |> Beaver.MLIR.Dialect.Registry.normalize_dialect_name()

      Logger.debug(
        "[Beaver] building Elixir module for dialect #{dialect} => #{dialect_module_name} (#{length(ops)})"
      )

      func_names =
        for op <- ops do
          func_name = Beaver.MLIR.Dialect.Registry.normalize_op_name(op)
          full_name = Enum.join([dialect, op], ".")

          def unquote(func_name)(ssa) do
            eval_ssa(%{ssa | op: unquote(full_name)})
          end

          defoverridable [{func_name, 1}]

          func_name
        end

      if length(func_names) != MapSet.size(MapSet.new(func_names)) do
        raise "duplicate op name found in dialect: #{dialect}"
      end
    end
  end

  defmacro define_modules(name) do
    quote bind_quoted: [d: name] do
      alias Beaver.MLIR.Dialect
      module_name = d |> Dialect.Registry.normalize_dialect_name()
      module_name = Module.concat([Beaver.MLIR.Dialect, module_name])

      ops = Dialect.Registry.ops(d)

      defmodule module_name do
        use Beaver.MLIR.Dialect,
          dialect: d,
          ops: ops
      end
    end
  end
end
