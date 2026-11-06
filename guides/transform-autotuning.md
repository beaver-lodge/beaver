# Transform execution

Construct Transform IR using `MLIR.Dialect.Transform` or parse a module with
`MLIR.Module.create!/2`. Apply a selected named sequence through
`MLIR.CAPI.mlirTransformApplyNamedSequenceWithDiagnostics/5`, passing the
context, payload operation, sequence operation, containing module operation,
and options from `mlirTransformOptionsCreate/0`. Destroy the options and schedule
after execution; the caller retains ownership of the payload.

`MLIR.Transform` still exposes standard transformation passes such as
`canonicalize/1`. Schedule selection, tuning, caching, and the retired
`Transform.execute/3` convenience layer belong to consumers. Serialize schedule
IR with `MLIR.Bytecode`; use `ActionTracing` and conversion profiling for
measurement. See the [core API migration](core-api-migration.md).
