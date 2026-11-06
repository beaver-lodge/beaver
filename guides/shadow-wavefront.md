# Shadow experiment API retirement

The experimental `Beaver.Shadow` harness and `mix shadow.*` tasks have been
retired. Applications own experiment inputs, evaluation, caching, and reports;
Beaver provides MLIR construction, passes, conversion, JIT, native tracing,
and bytecode. See the [core API migration](core-api-migration.md).
