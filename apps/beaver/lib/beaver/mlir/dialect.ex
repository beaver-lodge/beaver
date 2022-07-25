defmodule Beaver.MLIR.Dialect do
  alias Beaver.MLIR.Dialect

  require Logger

  for d <-
        Dialect.Registry.dialects()
        |> Enum.reject(fn x -> x in ~w{cf arith func builtin scf
          pdl_interp
          pdl
          linalg
          bufferization
          complex
          memref
          affine} end) do
    module_name = d |> Dialect.Registry.normalize_dialect_name()
    module_name = Module.concat([__MODULE__, module_name])

    defmodule module_name do
      use Beaver.MLIR.Dialect.Generator,
        dialect: d,
        ops: Dialect.Registry.ops(d)
    end

    module_name
  end

  def dialects() do
    for d <- Dialect.Registry.dialects() do
      module_name = d |> Dialect.Registry.normalize_dialect_name()
      Module.concat([__MODULE__, module_name])
    end
  end
end
