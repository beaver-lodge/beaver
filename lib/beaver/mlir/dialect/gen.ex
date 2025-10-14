require Beaver.MLIR.Dialect

explicitly_declared =
  ~w{cf arith func builtin tensor llvm spirv tosa linalg rocdl shape pdl_interp vector elixir memref}

for d <- Beaver.MLIR.Dialect.Registry.dialects(), d not in explicitly_declared do
  Beaver.MLIR.Dialect.define(d)
end
