# Beaver architecture

Beaver provides three layers of MLIR building blocks:

- **DSL:** `Beaver`, `Beaver.Env`, and `Beaver.SSA` express context-bound construction.
- **Utilities:** `Beaver.Walker`, `Beaver.Composer`, and `Beaver.Deferred` support
  inspection, pass composition, and contextual construction.
- **Bindings:** `Beaver.MLIR` exposes IR, dialects, passes, execution engines, and C APIs.

Applications own their pipeline, runtime, and artifact storage; see
[core API migration](core-api-migration.md).

## From Elixir syntax to MLIR

`Beaver.mlir/2` uses `Beaver.SSA.prewalk/2` to turn
`operation(arguments) >>> result_types` into a `%Beaver.SSA{}` carrying
arguments, result types, location, context, insertion point, and evaluator.
Dialect builders evaluate it through `Beaver.MLIR.Operation.eval_ssa/1`.

`module`, `region`, and `block` establish lexical construction scopes;
region bodies become callbacks. These scopes do not waive MLIR's dominance
or ownership rules: verify finished IR with `Beaver.MLIR.verify!/1`.
See [blocks](blocks.md) and [passes and patterns](passes-and-patterns.md).
Use MLIR text or bytecode for interchange, not a presumed Elixir-source round trip.

## Native boundary

Zig's `addTranslateC` translates the aggregate C API header; comptime reflection
constructs the NIF registry. Elixir APIs are compiled from declaration and
callback-bridge manifests, without generating wrapper source. Higher-level
structs and `Beaver.Walker`'s enumerable collections retain MLIR ownership rules.

## Context ownership

Contextual builders accept `ctx:` for eager construction; otherwise they return
`%Beaver.Deferred{}`. `Beaver.Deferred.resolve/2` materializes these values and
checks context ownership. Deferred construction is distinct from callbacks;
there is no process-global context.

Destroy owned modules before their context and never reuse destroyed handles.
The [README example](../README.md#build-and-verify-ir) shows explicit cleanup;
see `Beaver.Deferred` and `Beaver.MLIR.Context` for the detailed APIs.
