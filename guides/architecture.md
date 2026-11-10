# Beaver architecture

Beaver provides MLIR building blocks in three layers:

- **DSL:** `Beaver`, `Beaver.Env`, and `Beaver.SSA` translate Elixir syntax into
  context-bound MLIR construction.
- **Utilities:** `Beaver.Walker`, `Beaver.Composer`, and `Beaver.Deferred` support
  inspection, pass composition, and contextual construction.
- **Bindings:** `Beaver.MLIR` and its submodules expose MLIR operations, passes,
  dialects, execution engines, and the lower-level C API.

Applications choose their compiler pipeline, runtime, and artifact storage.
See the [core API migration guide](core-api-migration.md) for the boundary
between Beaver's primitives and application-owned orchestration.

## From Elixir syntax to MLIR

The SSA DSL keeps the structure of compiler code close to the IR it builds.
Inside `mlir ctx: ctx do`, with `Arith` aliased to
`Beaver.MLIR.Dialect.Arith`, this creates a module containing a constant:

<!-- beaver:test:module -->
```elixir
module do
  _v2 = Arith.constant(value: ~a{1 : i32}) >>> ~t<i32>
end
```

`Beaver.mlir/2` walks the quoted expression through `Beaver.SSA.prewalk/2`.
An expression of the form `operation(arguments) >>> result_types` constructs
a `%Beaver.SSA{}` containing the arguments, result types, source location,
context, insertion point, and evaluator. Dialect builders then evaluate that
description through `Beaver.MLIR.Operation.eval_ssa/1` and the operation
creation API.

`module`, `region`, and `block` establish lexical construction scopes;
`Beaver.Env` exposes their context and insertion point. Region bodies become
callbacks evaluated while the enclosing operation is constructed. The DSL
does not make arbitrary cross-region value references valid: MLIR's dominance
and ownership rules still apply. Always verify finished IR with
`Beaver.MLIR.verify!/1`. The [blocks guide](blocks.md) explains forward
references and dangling-block checks.

This is a construction DSL, not a promise that arbitrary MLIR can be printed
back into the original Elixir source. Use MLIR text or bytecode when exchanging
IR with other tooling.

## Passes and patterns

`Beaver.Composer` builds pass pipelines; passes operate on existing MLIR IR.
`Beaver.Pattern.defpat` translates declarative matches and rewrites into PDL
IR using Beaver's construction APIs. It is distinct from
`Beaver.Pattern.Native.defrewrite`, which installs an Elixir callback through
the native rewrite bridge.

See [passes and patterns](passes-and-patterns.md) for a complete PDL-backed
pass, and [native rewrite patterns](native-rewrite-patterns.md) for callback
lifetime and concurrency rules.

## Native boundary

Zig's Build System translates the aggregate MLIR C API header with
`addTranslateC`. Beaver reflects that namespace at comptime to construct the
NIF registry. The corresponding Elixir surface is compiled from
machine-readable declaration and callback-bridge manifests; no Zig or Elixir
wrapper source is generated.

Higher-level APIs wrap handles in Elixir structs. `Beaver.Walker` exposes
collections such as operation operands, results, and regions through
`Enumerable`, so callers can inspect IR with ordinary `Enum` functions.
Those conveniences do not change MLIR ownership or callback lifetimes.

## Context ownership

Higher-level contextual builders accept `ctx:` for eager construction.
Without it, they return an explicit `%Beaver.Deferred{}` that can be composed
into another builder or materialized with `Beaver.Deferred.resolve/2`.

Deferred values are not ordinary callback functions. The distinct type keeps
contextual construction separate from callback APIs and lets Beaver reject
cross-context entities before entering MLIR's native boundary. It does not
introduce a process-global MLIR context.

Destroy owned modules before their context, and do not use their borrowed
operations or values after destruction. The [README example](../README.md#build-and-verify-ir)
shows explicit cleanup with `try/after`. See `Beaver.Deferred` and
`Beaver.MLIR.Context` for the detailed APIs.
