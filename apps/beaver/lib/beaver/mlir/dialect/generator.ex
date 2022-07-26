alias Beaver.MLIR.Dialect

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
  module_name = Module.concat([Beaver.MLIR.Dialect, module_name])

  defmodule module_name do
    use Beaver.MLIR.Dialect,
      dialect: d,
      ops: Dialect.Registry.ops(d)
  end
end
