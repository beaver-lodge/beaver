defmodule Beaver.MLIR.Dialect.Generator do
  defmacro __using__(dialect: dialect, ops: ops) do
    quote(bind_quoted: [dialect: dialect, ops: ops]) do
      for op <- ops do
        func_name = Beaver.MLIR.Dialect.Registry.normalize_op_name(op)
        full_name = Enum.join([dialect, op], ".")

        def unquote(func_name)() do
          Beaver.MLIR.Operation.create(
            unquote(full_name),
            []
          )
          |> Beaver.MLIR.Operation.results()
        end

        def unquote(func_name)(args) do
          Beaver.MLIR.Operation.create(
            unquote(full_name),
            args
          )
          |> Beaver.MLIR.Operation.results()
        end

        # persistent the fullname so the caller module could access it
        @full_name full_name
        defmacro unquote(func_name)(args, do: ast_block) do
          quote(bind_quoted: [args: args, full_name: @full_name, ast_block: ast_block]) do
            if not is_list(args),
              do: raise("arguments passed to create operation should be a list")

            filler = fn -> ast_block end

            Beaver.MLIR.Operation.create(
              full_name,
              args ++
                [
                  regions: filler
                ]
            )
            |> Beaver.MLIR.Operation.results()
          end
        end
      end
    end
  end
end
