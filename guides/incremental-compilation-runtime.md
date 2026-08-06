# Incremental MLIR compilation

`Beaver.MLIR.CompilationRuntime` shortens the edit–compile–reload loop for NIFs
and other native extensions. A long-running BEAM application owns one LLVM
thread pool, and every default MLIR context leases that pool instead of creating
another one.

The runtime caches transformed MLIR bytecode rather than a live context or JIT.
This makes the handoff explicit: the same validated artifact can be loaded into
a new execution engine or emitted as an object file.

```elixir
alias Beaver.MLIR

source = File.read!("native/kernel.mlir")

compile_opts = [
  pipeline: "convert-func-to-llvm,convert-arith-to-llvm",
  target: %{triple: "aarch64-apple-darwin", features: "+neon"},
  schema_version: "my_dynamic_dialect/v3",
  cache: {:file, ".beaver/cache"}
]

artifact = MLIR.CompilationRuntime.compile!(source, compile_opts)

jit = MLIR.CompilationRuntime.jit!(artifact)
MLIR.CompilationRuntime.emit_object!(artifact, "_build/native/kernel.o")

# After editing native/kernel.mlir, compile and load the replacement first.
next_artifact =
  "native/kernel.mlir"
  |> File.read!()
  |> MLIR.CompilationRuntime.compile!(compile_opts)

next_jit = MLIR.CompilationRuntime.jit!(next_artifact)
MLIR.ExecutionEngine.destroy(jit)
jit = next_jit
```

The artifact key contains the source digest and structural hash, Beaver's LLVM
version, pipeline identity, target configuration, dynamic dialect/schema
version, and requested bytecode version. Changing any of these inputs is a
cache miss. Stored bytecode is checksummed and its metadata is revalidated
before use; read, write, or validation failure falls back to a normal compile.

For a custom Elixir pass or another runtime function, provide a stable version
because a closure is not a deterministic cache identity:

```elixir
MLIR.CompilationRuntime.compile!(source,
  pipeline: [MyPass],
  pipeline_version: {MyPass, 4}
)
```

Use `cache: :memory` for process-local iteration, or `cache: {:file, path}` to
reuse artifacts across BEAM restarts. Invalidate one lookup key or the whole
backend explicitly:

```elixir
MLIR.CompilationRuntime.invalidate(:memory, artifact.metadata.lookup_key)
MLIR.CompilationRuntime.invalidate({:file, ".beaver/cache"})
```

Compilation emits events below `[:beaver, :mlir, :compilation]` for cache
lookup/hit/miss/failure, parse, transform, serialization, codegen, JIT load,
object emission, and execution. Measurements use native time units. If the
`:telemetry` package is not installed, pass a callback with
`telemetry: fn event, measurements, metadata -> ... end`.

Run the cold/warm comparison with:

```sh
mix run bench/incremental_compilation.exs
```
