# Ex ownership and conversion boundaries

Ex declarations have three owners. The `Beaver.MLIR.Dialect.Ex` facade composes
them into the existing `ex` namespace, so consumers keep the same operation
names, type names, builders, operand names and segment conventions.

| Owner | Responsibility | Examples |
| --- | --- | --- |
| `Ex.Language` | Values, calls, collections and structured language control | `case`, `clause`, `try`, `integer_mul`, `tuple`, `call` |
| `Ex.Execution` | Scheduler state, continuations, mailbox scans and reductions | `cont_save`, `receive_cont_save`, `reduction_tick`, `process_wait` |
| `Ex.Runtime` | Word representation, closure layout and host handles | `box`, `unbox`, `make_fun_with_signature`, `runtime_create`, `term_export` |

These are schema fragments, not three independently loadable MLIR dialects.
Load the complete dialect with `Beaver.Slang.load(ctx, Ex)`. Each declaration
registers its owner when compiled; the catalog is derived from declarations,
and duplicate ownership fails compilation. The facade exposes `layers/0`,
`operations/0`, `operations/1` and `layer/1` for providers and diagnostics.

## Check a boundary

```elixir
alias Beaver.MLIR.Dialect.Ex

Ex.verify_layers!(source_ir, :language)
Ex.verify_layers!(scheduled_ir, [:language, :execution])
```

`verify_layers!/2` first runs MLIR verification, then walks all nested regions
and rejects Ex operations whose owner is not explicitly allowed. Unknown Ex
names also reject. Other dialects remain subject to normal MLIR verification;
this helper is not a whitelist of all possible operations or a proof of source
equivalence. It returns the original module or operation without changing it.

## Convert selected owners

The default reference conversion is compatible with existing callers:

```elixir
alias Beaver.MLIR.Conversion.{Ex, Plan}
converted = Plan.run!(Ex.plan(), module)
```

Providers can convert one owner while retaining the others:

```elixir
module = Plan.run!(Ex.plan(layers: :execution), module)
module = Plan.run!(Ex.plan(layers: :runtime), module)
module = Ex.finish!(module)
```

Layer selection controls operation legality and callback registration. The
frozen native scalar seed is unchanged. Intermediate layer plans default to
`build_materializations: false`, allowing explicit
`builtin.unrealized_conversion_cast` bridges between `!ex.term` and `i64`.
`finish!/1` converts remaining Ex operations, then runs
`reconcile-unrealized-casts`; unresolved bridges fail instead of reaching the
target backend. A layer plan alone is an intermediate artifact.

Language patterns still require the existing preparation passes:
`Ex.MaterializeBoundVariables.run!/1` and `Ex.ExpandCase.run!/1` (dialect
modules). Selection does not manufacture missing pattern or exception rules.
Conversion continues to mutate the supplied IR, following `Conversion.Plan`.

## Runtime identity and effects

`Beaver.MLIR.Conversion.Ex.RuntimeABI` owns the reference runtime symbol
aliases and function signatures, including pointer and callback arguments.
`Conversion.Ex.term_intrinsic_symbols/0` remains a compatibility delegate.
Unknown runtime symbols have no inferred default signature. This ABI belongs
to the original Batata Zig runtime; the current wavefrontier/batata C runtime
is a separate backend and is not integrated by this reorganization.

Scalar literals, add/subtract/multiply, comparisons and representation-only
box/unbox operations declare memory freedom and speculatability. Runtime
calls, allocation, integer division, exceptions and scheduler operations keep
conservative behavior. Language ownership alone never implies purity.

The schema identity now hashes the facade plus every declaration fragment and
the shared declaration macro. Existing content-bound provider receipts must
be rebuilt for the new identity even though operation signatures are retained.
The C++ Stage 0 manifest and its identity are unchanged.

## Semantic work this boundary enables

The split preserves current semantics and restrictions: scalar case expansion,
two-region `try`, fixed closure slots, integer-valued host handles and the
legacy continuation convention remain. The following require new contracts
and lowering implementations rather than merely moving declarations:

- Patterns, guard evaluation and failed-branch binding scope as structured IR.
- Separate rescue/else/after paths and their constraints on tail calls.
- Checked suspension plans carrying costs, live values and GC roots.
- Logical closure environments and typed host capabilities independent of ABI.

New language declarations belong in `Language`; scheduling decisions belong
in `Execution`; representation and host conventions belong in `Runtime` and
the backend's ABI module. Each addition should define its verifier, effects
and conversion obligations before being admitted by a production provider.
