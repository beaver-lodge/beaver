defmodule Beaver.MLIR.Dialect do
  alias Beaver.MLIR.Dialect

  require Logger

  module_names =
    for d <-
          Dialect.Registry.dialects(query: true)
          |> Enum.reject(fn x -> x in ~w{cf arith func builtin scf
          pdl_interp
          pdl
          linalg
          bufferization
          complex
          memref
          affine} end) do
      module_name = d |> Dialect.Registry.normalize_dialect_name()
      Logger.debug("building Elixir module for dialect #{d} => #{module_name}")
      module_name = Module.concat([__MODULE__, module_name])

      defmodule module_name do
        use Beaver.MLIR.Dialect.Generator,
          dialect: d,
          ops: Dialect.Registry.ops(d, query: true)
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

  def dialects() do
    for module_name <- @module_names do
      module_name
    end
  end
end
