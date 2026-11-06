# Core API migration

The core reduction from 25,000 to fewer than 20,000 code lines removes the
experimental orchestration APIs below. This is an intentional API change, in
addition to deduplication of Slang builders, conversion plans, and callback
encoding. It does not provide compatibility stubs for the retired APIs.

The count uses the same implementation scope before and after:

```sh
cloc lib native --skip-uniqueness --quiet
```

Blank lines and comments are excluded. Implementation has not been moved into
tests, build output, another dependency, or a different directory to meet the
limit. Kinda remains pinned to the same revision.

## Retired interfaces

| Retired API | Supported replacement or migration |
| --- | --- |
| `Beaver.MLIR.TypeConverterCallback` | `Beaver.MLIR.TypeConverter.create(conversion: callback)` and `TypeConverter.convert/2`. |
| `Beaver.MLIR.CompilationPlan` and its declaration macro | Compose passes with `Beaver.Composer`; use `Beaver.MLIR.Conversion.Plan` for dialect conversion. |
| `CompilationRuntime.compile!`, `jit!`, `emit_object!`, and cache operations | Use `MLIR.Module`, `Composer`, `MLIR.Bytecode`, and `MLIR.ExecutionEngine` directly. `CompilationRuntime.llvm_revision/0` remains available. |
| `Beaver.MLIR.CompilationCache` and its Memory/File backends; compilation Artifact/CacheKey structs | Applications own cache storage, keys, invalidation, and artifact metadata. |
| <code>Beaver.MLIR.Transform.Schedule</code> discovery, enumeration, resolution, serialization, and cache identity; its Option/Choice/Constraint/Analysis/Resolved structs | Author selected Transform IR and call `MLIR.Transform.execute/3`; serialize with `MLIR.Bytecode`. The internal parsing/execution bridge remains. |
| `Beaver.MLIR.Transform.Schedule.DSL`, `Tuner`, `Resolver`, and `Solver` | Use native Transform dialect operations; applications own search, scoring, and solver integration. |
| `Beaver.Shadow.*` and the `mix shadow.*` tasks | Applications own experiment orchestration and reports. Native Transform, Triton, CUDA, ActionTracing, and conversion profiling remain. |
| `Beaver.MLIR.Triton.LayoutAudit` and `PipelineEvidence` | Inspect IR with `Beaver.Walker` and collect native action traces or conversion profiles. `Triton.PipelinePlan` remains. |
| `Beaver.Wasm.TermABI` | Runtime implementers own their ABI manifest. `Beaver.Wasm` and `Wasm.Binary` remain. |
| `Conversion.Ex.RuntimeABI`, `term_intrinsic_symbols/0`, old Ex Zig runtime/continuation operations and tagged-word lowering | `Conversion.Ex.CRuntime`: raw `ex.btt_*` or SSA `ex.value_*` calls target Batata C ABI 12, including chunked imports and cross-guest copies. Language operations still require a provider; old handles/bytecode are incompatible. See [the runtime boundary](dialect-conversion.md#ex-provider-boundary). |

The incremental compilation benchmark and tests dedicated to retired APIs are
removed with their implementations. Tests for retained callbacks, native
Transform execution, Slang, conversion, and bytecode round trips remain.

## Compatibility scope

The removal audit covered Beaver itself and local Batata, Charms, Manx, Compas,
and Beaver IR source/test/configuration files. None of those consumers uses the
retired entry points. Batata's use of `CompilationRuntime.llvm_revision/0` is
explicitly preserved. This audit cannot establish usage by external downstream
projects; those projects should check the table before updating.

The native C API declarations, callback bridge metadata, and hand-written NIF
export names/arities are unchanged. Existing MLIR construction, Beaver's main
DSL, Slang, rewrite patterns, conversion plans, JIT, Transform execution, native
tracing, Triton pipeline construction, and CUDA operations remain supported.

See [direct compilation](incremental-compilation-runtime.md) and
[Transform execution](transform-autotuning.md) for migration examples.
