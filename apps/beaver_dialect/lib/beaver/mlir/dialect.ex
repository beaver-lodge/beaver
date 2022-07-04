defmodule Beaver.MLIR.Dialect do
  alias Beaver.MLIR.Dialect
  Application.ensure_all_started(:beaver_capi)

  for d <-
        Dialect.Registry.dialects()
        |> Enum.reject(fn x -> x in ~w{cf arith func builtin} end) do
    module_name = d |> Dialect.Registry.normalize_dialect_name()
    IO.puts("building Elixir module for #{d} => #{module_name}")
    module_name = Module.concat([__MODULE__, module_name])

    Task.async(fn ->
      defmodule module_name do
        use Beaver.MLIR.Dialect.Generator,
          dialect: d,
          ops: Dialect.Registry.ops(d)
      end
    end)
  end
  |> Task.await_many(:infinity)
end
