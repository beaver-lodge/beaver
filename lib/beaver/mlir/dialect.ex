defmodule Beaver.MLIR.Dialect do
  alias Beaver.MLIR.Dialect

  require Logger

  defmacro __using__(opts) do
    dialect = Keyword.fetch!(opts, :dialect)
    ops = Keyword.fetch!(opts, :ops)

    quote(bind_quoted: [dialect: dialect, ops: ops]) do
      require Logger
      dialect_module_name = dialect |> Beaver.MLIR.Dialect.Registry.normalize_dialect_name()

      Logger.debug(
        "[Beaver] building Elixir module for dialect #{dialect} => #{dialect_module_name} (#{length(ops)})"
      )

      op_module_names =
        for op <- ops do
          func_name = Beaver.MLIR.Dialect.Registry.normalize_op_name(op)
          full_name = Enum.join([dialect, op], ".")

          def unquote(func_name)(%Beaver.DSL.SSA{evaluator: evaluator} = ssa)
              when is_function(evaluator, 2) do
            evaluator.(unquote(full_name), ssa)
          end

          Module.concat([
            Beaver.MLIR.Dialect,
            dialect_module_name,
            Beaver.MLIR.Dialect.Registry.op_module_name(op)
          ])
        end

      if length(op_module_names) != MapSet.size(MapSet.new(op_module_names)) do
        raise "duplicate op name found in dialect: #{dialect}"
      end

      @op_module_names op_module_names

      def op_module_names() do
        []
      end

      defoverridable op_module_names: 0

      def __ops__() do
        @op_module_names ++ op_module_names()
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

      Beaver.MLIR.Dialect.define_op_modules(module_name, d, Dialect.Registry.ops(d))
    end
  end

  defmacro define_op_modules(dialect_module, dialect, ops) do
    quote bind_quoted: [module_name: dialect_module, d: dialect, ops: ops] do
      for op <- ops do
        full_name = Enum.join([d, op], ".")

        op_module_name =
          Module.concat([
            module_name,
            Beaver.MLIR.Dialect.Registry.op_module_name(op)
          ])

        defmodule op_module_name do
          use Beaver.DSL.Op.Prototype, op_name: unquote(full_name)
        end
      end
    end
  end
end
