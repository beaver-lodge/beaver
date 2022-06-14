defmodule Beaver.MLIR.Dialect.Generator do
  defmacro __using__(dialect: dialect, ops: ops) do
    quote(bind_quoted: [dialect: dialect, ops: ops]) do
      for op <- ops do
        func_name = Beaver.MLIR.Dialect.Registry.normalize_op_name(op)
        full_name = Enum.join([dialect, op], ".")

        def unquote(func_name)(args) do
          Beaver.MLIR.Operation.create(
            unquote(full_name),
            args
          )
          |> Beaver.MLIR.Operation.results()
        end
      end
    end
  end
end
