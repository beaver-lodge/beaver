defmodule Beaver.MLIR.Dialect do
  alias Beaver.MLIR.Dialect

  require Logger

  module_names =
    for d <-
          Dialect.Registry.dialects()
          |> Enum.reject(fn x -> x in ~w{cf arith func builtin} end) do
      module_name = d |> Dialect.Registry.normalize_dialect_name()
      Logger.debug("building Elixir module for dialect #{d} => #{module_name}")
      module_name = Module.concat([__MODULE__, module_name])

      defmodule module_name do
        use Beaver.MLIR.Dialect.Generator,
          dialect: d,
          ops: Dialect.Registry.ops(d)
      end

      module_name
    end

  @module_names module_names ++
                  [
                    Beaver.MLIR.Dialect.CF,
                    Beaver.MLIR.Dialect.Arith,
                    Beaver.MLIR.Dialect.Func,
                    Beaver.MLIR.Dialect.Builtin
                  ]

  @on_load :load_dialects

  def load_dialects() do
    for module_name <- @module_names do
      Code.ensure_loaded!(module_name)
    end

    :ok
  end
end
