defmodule Beaver.MLIR.Dialect.Generator do
  defmacro __using__(opts) do
    dialect = Keyword.fetch!(opts, :dialect)
    ops = Keyword.fetch!(opts, :ops)
    skips = Keyword.get(opts, :skips, [])

    quote(bind_quoted: [dialect: dialect, ops: ops, skips: skips]) do
      module_names =
        for op <- ops do
          func_name = Beaver.MLIR.Dialect.Registry.normalize_op_name(op)
          full_name = Enum.join([dialect, op], ".")

          module_name = dialect |> Beaver.MLIR.Dialect.Registry.normalize_dialect_name()

          module_name =
            Module.concat([
              Beaver.MLIR.Dialect,
              module_name,
              Beaver.MLIR.Dialect.Registry.op_module_name(op)
            ])

          if op not in skips do
            defmodule module_name do
              use Beaver.DSL.Op.Prototype, op_name: unquote(full_name)
            end

            require Beaver.Env

            def unquote(func_name)(%Beaver.DSL.SSA{} = ssa) do
              Beaver.MLIR.Operation.create(
                unquote(full_name),
                ssa
              )
              |> Beaver.MLIR.Operation.results()
            end

            # persistent the fullname so the caller module could access it
            @full_name full_name
            defmacro unquote(func_name)(%Beaver.DSL.SSA{} = ssa, do: ast_block) do
              quote(bind_quoted: [ssa: ssa, full_name: @full_name, ast_block: ast_block]) do
                Beaver.MLIR.Operation.create(full_name, ssa)
                |> Beaver.MLIR.Operation.results()
              end
            end
          end

          module_name
        end

      if length(module_names) != MapSet.size(MapSet.new(module_names)) do
        raise "duplicate op name found in dialect: #{dialect}"
      end

      @module_names module_names

      def ops() do
        for module_name <- @module_names do
          module_name
        end
      end
    end
  end
end
