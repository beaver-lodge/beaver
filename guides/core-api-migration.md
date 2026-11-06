# Core API migration

The reductions to 20,000 and then 15,000 code lines combine shared builders,
callback encoding, and header-driven ENIF exports with retirement of unused
experimental helpers. Retired APIs have no compatibility stubs.

Both counts use `cloc lib native --skip-uniqueness --quiet`, excluding blanks
and comments. Implementation has not moved to tests, build output, dependencies,
or another directory. The Kinda revision is unchanged.

| Retired API | Replacement |
| --- | --- |
| `TypeConverterCallback` | `MLIR.TypeConverter` callbacks. |
| `CompilationPlan`, compilation cache/artifact structs, `CompilationRuntime` orchestration | `Composer`, `Conversion.Plan`, `Module`, `Bytecode`, and `ExecutionEngine`; consumers own caching. `CompilationRuntime.llvm_revision/0` remains. |
| `Transform.Schedule`, DSL, Tuner, Resolver, Solver, `Transform.execute` / `apply_named_sequence` and type helpers | Transform dialect operations and `CAPI.mlirTransformApplyNamedSequenceWithDiagnostics`; `Type.get` for handle types. Standard `MLIR.Transform` passes remain. |
| `Transform.composite_fixed_point` | Native textual pass pipelines through `Composer`. |
| `Beaver.Shadow.*`, `mix shadow.*` | Consumer-owned experiments and reporting. |
| `Triton.LayoutAudit`, `PipelineEvidence`, `PipelinePlan`, pipeline/profile helpers | Native Triton passes through `Composer`; `Walker`, `ActionTracing`, conversion profiling. |
| `MLIR.CUDA`, native CUDA bootstrap NIFs | Consumer-owned CUDA runtime loading. GPU dialect compilation and LLVM-to-PTX remain. |
| `Beaver.Wasm`, `Wasm.Binary`, `Wasm.TermABI` | Consumer-owned Wasm linking, packaging, and runtime ABI. Ex runtime intrinsic symbols remain. |
| `Beaver.Pattern`, `Beaver.Pattern.Native`, descriptors and `Conversion.Plan.add_pattern` | Four-argument `RewritePattern` callbacks, `RewritePatternSet.add/4`, or `Plan.add_pattern_population/3`; construct native PDL operations directly. |
| Custom dynamic trait callbacks, `TransformOpInterface`, `PatternDescriptorOpInterface`, transform memory-effect helpers | Native extension interfaces. Built-in traits, memory-effect and speculation callbacks remain. |
| `InferType`, `InferShapedType`, `Inference.Request` collectors | Operation result inference (`>>> :infer`) remains. Standalone callback collectors are no longer bridged. |
| `Context.with_transient_scope` and transient-scope adapters | Use separate contexts to bound allocations; native context CAPI remains available where LLVM provides it. |
| `Location` named/fused/call-site/range and inspection helpers | Native location CAPI; `file`, `from_env`, `unknown`, `of`, and context-checked `resolve` remain. |
| `DominanceInfo` / `PostDominanceInfo` helpers | Native dominance CAPI; handle types remain. |
| LLVM type/attribute helpers and DLTI builders | `Type.get`, `Attribute.get`, generated dialect operations, or CAPI. `LLVM.func` remains. |
| `SCF.if_` / `for_`, `Tensor.Slice` helpers, `Vector.transfer_read_` / `in_bounds` | Generated `SCF`, `Tensor`, and `Vector` operations with explicit regions and attributes. |
| Bufferization pipeline helpers, `Linalg.llvm_pipeline` / `lower_to_llvm!` | Native textual pipelines through `Composer`; dialect operations and structured Linalg construction remain. |

The audit covered local Batata (the older Elixir checkout), Charms, Manx,
Compas, and Beaver IR source, tests, and configuration. Their required entry
points are retained. External consumers must check this table before updating.
Native exports and callback metadata for retired bridges are removed or marked
pending; this is an intentional API change, not a promise of universal ABI
compatibility.

See [direct compilation](incremental-compilation-runtime.md),
[callback rewrites](native-rewrite-patterns.md), and
[Transform execution](transform-autotuning.md).
