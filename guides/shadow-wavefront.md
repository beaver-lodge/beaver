# Shadow experiment API retirement

The `Beaver.Shadow` experiment harness, GPU/Wasm wrappers, receipts, reports,
and `mix shadow.*` tasks have been retired. They had no callers in the audited
local consumer projects and duplicated application-level orchestration.

Beaver retains native Transform execution, Triton pipeline construction, CUDA
operations, Wasm translation, action tracing, conversion profiling, and MLIR
bytecode serialization. Applications can compose these primitives while owning
their experiment inputs, evaluation, cache, and reporting policy.

See the [core API migration](core-api-migration.md),
[Transform execution](transform-autotuning.md), and
[direct compilation](incremental-compilation-runtime.md) for the retained API
and migration paths. This page preserves the old documentation link; its
former Shadow examples no longer apply.
