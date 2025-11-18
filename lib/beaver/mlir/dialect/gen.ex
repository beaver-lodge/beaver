require Beaver.MLIR.Dialect

explicitly_declared =
  for p <- Path.wildcard("lib/beaver/mlir/dialect/*.ex") do
    p |> Path.basename(".ex")
  end

for d <- Beaver.MLIR.Dialect.Registry.dialects(), d not in explicitly_declared do
  Beaver.MLIR.Dialect.define(d)
end
