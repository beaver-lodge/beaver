defmodule Beaver.MLIR.Dialect do
  @moduledoc """
  This module defines macro to generate code for an MLIR dialect.
  You might override `eval_ssa/1` function to introduce your custom op generation
  """

  use Kinda.ResourceKind, forward_module: Beaver.Native

  @callback eval_ssa(Beaver.SSA.t()) :: any()
  defmacro __using__(opts) do
    dialect = Keyword.fetch!(opts, :dialect)
    ops = Keyword.fetch!(opts, :ops)

    quote(bind_quoted: [dialect: dialect, ops: ops]) do
      @behaviour Beaver.MLIR.Dialect
      @doc false
      def eval_ssa(%Beaver.SSA{evaluator: evaluator} = ssa) do
        evaluator.(ssa)
      end

      defoverridable eval_ssa: 1

      dialect_module_name = dialect |> Beaver.MLIR.Dialect.Registry.normalize_dialect_name()

      func_names =
        for op <- ops do
          func_name = Beaver.MLIR.Dialect.Registry.normalize_op_name(op)
          full_name = Enum.join([dialect, op], ".")

          @doc (case(Beaver.MLIR.ODS.Dump.lookup(full_name)) do
                  {:ok, found} ->
                    Beaver.MLIR.ODS.Dump.gen_doc(found)

                  _ ->
                    "`#{full_name}`"
                end)
          @file full_name
          def unquote(func_name)(ssa) do
            eval_ssa(%Beaver.SSA{ssa | op: unquote(full_name)})
          end

          @doc false
          def unquote(func_name)() do
            unquote(full_name)
          end

          defoverridable [{func_name, 1}]

          func_name
        end

      if length(func_names) != MapSet.size(MapSet.new(func_names)) do
        raise "duplicate op name found in dialect: #{dialect}"
      end
    end
  end

  defmacro define(name) do
    quote bind_quoted: [d: name] do
      alias Beaver.MLIR.Dialect
      module_name = d |> Dialect.Registry.normalize_dialect_name()
      module_name = Module.concat([Beaver.MLIR.Dialect, module_name])

      ops = Dialect.Registry.ops(d)

      defmodule module_name do
        @file d
        use Beaver.MLIR.Dialect,
          dialect: d,
          ops: ops
      end
    end
  end
end
