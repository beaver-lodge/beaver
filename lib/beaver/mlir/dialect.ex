defmodule Beaver.MLIR.Dialect do
  @moduledoc """
  This module defines macro to generate code for an MLIR dialect.
  """
  alias Beaver.MLIR.Dialect

  require Logger

  @callback eval_ssa(String.t(), Beaver.DSL.SSA.t()) :: any()
  defmacro __using__(opts) do
    dialect = Keyword.fetch!(opts, :dialect)
    ops = Keyword.fetch!(opts, :ops)

    quote(bind_quoted: [dialect: dialect, ops: ops]) do
      @behaviour Beaver.MLIR.Dialect

      def eval_ssa(full_name, %Beaver.DSL.SSA{evaluator: evaluator} = ssa)
          when is_function(evaluator, 2) do
        evaluator.(full_name, ssa)
      end

      defoverridable eval_ssa: 2
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
            eval_ssa(unquote(full_name), ssa)
          end

          defoverridable [{func_name, 1}]

          func_name
        end

      if length(func_names) != MapSet.size(MapSet.new(func_names)) do
        raise "duplicate op name found in dialect: #{dialect}"
      end
    end
  end

  def dialects() do
    for d <- Dialect.Registry.dialects() do
      module_name = d |> Dialect.Registry.normalize_dialect_name()
      Module.concat([__MODULE__, module_name])
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
