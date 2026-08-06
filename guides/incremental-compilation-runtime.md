# Incremental MLIR compilation

`Beaver.MLIR.CompilationRuntime` shortens the edit–compile–reload loop for NIFs
and other native extensions. A long-running BEAM application owns one LLVM
thread pool, and every default MLIR context leases that pool instead of creating
another one.

The runtime caches transformed MLIR bytecode rather than a live context or JIT.
This makes the handoff explicit: the same validated artifact can be loaded into
a new execution engine or emitted as an object file.

## Declarative compilation plans

`Beaver.MLIR.CompilationPlan` closes a compilation configuration into a reusable,
inspectable, cache-stable declaration above `Composer`, `Transform.Schedule`, and
`CompilationRuntime`.

```elixir
defmodule MyNativeExtension do
  import Beaver.MLIR.CompilationPlan

  defcompiler kernel_compiler do
    Beaver.MLIR.CompilationPlan.new(
      pipeline: "convert-func-to-llvm,convert-arith-to-llvm",
      target: %{triple: "aarch64-apple-darwin", features: "+neon"},
      schema_version: "my_dynamic_dialect/v3",
      telemetry_metadata: %{compiler: :kernel}
    )
  end
end

alias Beaver.MLIR

source = File.read!("native/kernel.mlir")
plan = MyNativeExtension.kernel_compiler()
cache = {:file, ".beaver/cache"}

artifact = MLIR.CompilationRuntime.compile!(source, plan, cache: cache)

jit = MLIR.CompilationRuntime.jit!(artifact)
MLIR.CompilationRuntime.emit_object!(artifact, "_build/native/kernel.o")

# After editing native/kernel.mlir, compile and load the replacement first.
next_artifact =
  "native/kernel.mlir"
  |> File.read!()
  |> MLIR.CompilationRuntime.compile!(plan, cache: cache)

next_jit = MLIR.CompilationRuntime.jit!(next_artifact)
MLIR.ExecutionEngine.destroy(jit)
jit = next_jit
```

`Beaver.MLIR.CompilationPlan.declaration/1` contains only deterministic metadata,
and `Beaver.MLIR.CompilationPlan.identity/1` hashes that projection without creating an MLIR
context. Cache backends, borrowed contexts, telemetry callbacks, and LLVM revision
overrides remain runtime options and are not stored in the declaration.

The artifact key contains the source digest and structural hash, Beaver's LLVM
version, plan identity (`plan_id`), target configuration, dynamic dialect/schema
version, and requested bytecode version. Changing any of these inputs is a
cache miss. Stored bytecode is checksummed and its metadata is revalidated
before use; read, write, or validation failure falls back to a normal compile.

For a custom Elixir pass or another runtime function, provide a stable version
because a closure is not a deterministic cache identity:

```elixir
plan =
  Beaver.MLIR.CompilationPlan.new()
  |> Beaver.MLIR.CompilationPlan.add_pass(MyPass, version: {MyPass, 4})

MLIR.CompilationRuntime.compile!(source, plan)
```

Callback passes use the same Composer tuple and must also carry a stable version:

```elixir
plan =
  Beaver.MLIR.CompilationPlan.new()
  |> Beaver.MLIR.CompilationPlan.add_pass(
    {"lower-runtime", "builtin.module", &lower_runtime/1},
    version: 3
  )
```

The callback itself remains executable runtime payload; only its pass argument,
root operation, and explicit version enter the declaration. The legacy keyword API
continues to accept `:pipeline_version` for compatibility, but plans version each
module or callback step so changes to surrounding pass and nested-scope data cannot
be hidden by one manually maintained pipeline version.

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
