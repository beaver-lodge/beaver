# Direct MLIR compilation and artifact reuse

The experimental `CompilationPlan`, compilation cache backends, and
`CompilationRuntime` orchestration API have been retired. Only
`Beaver.MLIR.CompilationRuntime.llvm_revision/0` remains, for build identity.
See the [core API migration](core-api-migration.md) for the complete removal list.

Use the underlying compilation and serialization APIs directly. This example
owns a context and module, runs a pass pipeline, and returns portable bytecode:

```elixir
alias Beaver.{Composer, MLIR}

source = """
module {
  func.func @add(%lhs: i32, %rhs: i32) -> i32 {
    %result = arith.addi %lhs, %rhs : i32
    return %result : i32
  }
}
"""

context = MLIR.Context.create()

bytecode =
  try do
    module = MLIR.Module.create!(source, ctx: context)

    try do
      module
      |> Composer.append("convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts")
      |> Composer.run!()
      |> MLIR.Bytecode.write!()
    after
      MLIR.Module.destroy(module)
    end
  after
    MLIR.Context.destroy(context)
  end
```

To reuse the artifact, parse `bytecode` with `MLIR.Module.create!/2` in a fresh
context and create an engine with `MLIR.ExecutionEngine.create!/2`. For object
output, create the engine with `object_dump: true` and call
`MLIR.ExecutionEngine.emit_object!/2`. Destroy the engine before destroying its
module and context.

Applications now own cache persistence and invalidation. Include source,
pipeline configuration, LLVM revision, target, and dynamic dialect definitions
in the application's cache identity, and validate artifacts before reuse.
There is no automatic cache or compilation telemetry replacement. Existing
execution-engine telemetry and native action tracing remain available.
