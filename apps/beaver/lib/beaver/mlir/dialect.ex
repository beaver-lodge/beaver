defmodule Beaver.MLIR.Dialect do
  alias Beaver.MLIR.Dialect
  Application.ensure_all_started(:beaver_mlir)

  for d <-
        Dialect.Registry.dialects() |> Enum.reject(fn x -> x in ~w{cf pdl arith func builtin} end) do
    module_name = d |> Dialect.Registry.normalize_dialect_name()
    module_name = Module.concat([__MODULE__, module_name])

    defmodule module_name do
      use Beaver.MLIR.Dialect.Generator,
        dialect: d,
        ops: Dialect.Registry.ops(d)
    end
  end
end
