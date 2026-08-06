# Native Rewrite Patterns

Beaver exposes two deliberately separate rewrite backends:

1. `Beaver.Pattern.defpat` builds declarative PDL IR.
2. `Beaver.Pattern.Native.defrewrite` builds a descriptor for an Elixir
   `MatchAndRewrite` callback backed by Beaver's existing native bridge.

There is no implicit fallback between them.

## Declaring a callback rewrite

```elixir
defmodule MyRewrites do
  use Beaver
  import Beaver.Pattern.Native

  alias Beaver.MLIR.Dialect.Arith

  defrewrite fold_add(operation, rewriter, state),
    root: Arith.addi(),
    operands: [lhs, rhs],
    results: [_result],
    benefit: 2 do
    # Inspect lhs and rhs, then mutate IR through the borrowed rewriter.
    # Return :no_match when the extra conditions for this rewrite do not hold.
    _base = MLIR.PatternRewriter.as_base(rewriter)
    _context = MLIR.context(operation)
    _state = state
    {lhs, rhs}
    :no_match
  end
end
```

`defrewrite` requires an explicit root operation. Its optional `:operands`,
`:results`, and `:attributes` entries are Elixir patterns. Operand or result
shape mismatches and missing attributes become no-match results without
evaluating the body. The three callback arguments must be variables; destructure
callback state inside the body. The root is installed as MLIR's native
operation filter, so unrelated operation names do not enter the callback.

The declaration above generates `fold_add/0` and `fold_add/1`. Both return a
`Beaver.Pattern.Native.Descriptor`; the one-arity builder accepts `:root`,
`:benefit`, `:init_state`, `:construct`, and `:destruct` overrides.

Use a descriptor in the list-based rewrite API:

```elixir
MLIR.Rewrite.apply_patterns!(ir, [
  MyRewrites.fold_add(benefit: 5)
])
```

Or add it to a mutable set:

```elixir
set = MLIR.RewritePatternSet.create(ctx)
MLIR.RewritePatternSet.add(set, MyRewrites.fold_add(), ctx: ctx)
```

## Match and state outcomes

A body must return one of these outcomes:

- `:ok` becomes `{:ok, state}`.
- `:no_match` becomes `{:error, state}`.
- `{:ok, new_state}` reports a successful rewrite and advances state.
- `{:error, new_state}` reports no-match and still advances state.

The `:init_state`, `:construct`, and `:destruct` options use the same lifecycle
as `Beaver.MLIR.RewritePattern`. Exceptions and unsupported outcomes are caught
at that existing callback boundary, logged on the BEAM side, and reported to
MLIR as pattern failure.

A successful callback must actually update the IR through
`Beaver.MLIR.PatternRewriter` or `Beaver.MLIR.RewriterBase`. Reporting success
without a tracked mutation can repeatedly reactivate the same pattern and keep
the greedy driver from converging.

## Native callbacks versus PDL

| Property | Native callback (`defrewrite`) | PDL (`defpat`) |
| :--- | :--- | :--- |
| Execution | Elixir callback reached through the native bridge | Declarative PDL interpreted by MLIR |
| Expressivity | Arbitrary Elixir checks, services, and callback state | PDL operations and constraints |
| Per-match cost | Includes native-to-BEAM callback and message dispatch | Remains inside MLIR's PDL execution path |
| Portability | Requires Beaver and the BEAM runtime | Pattern IR can be printed or serialized for MLIR tooling |
| Failure | Elixir exception is logged and becomes pattern failure | MLIR/PDL diagnostic and failure semantics |

Choose PDL when the match is naturally declarative or callback overhead would
dominate. Choose a native rewrite when the condition or transformation needs
Elixir code, state, or integrations that PDL cannot express cleanly.

## Borrowing and concurrency

The operation, rewrite-pattern handle, and `PatternRewriter` delivered to the
callback are borrowed. They are valid only for the dynamic extent of that
callback. Do not send them to another process, store them in callback state, or
use them after the body returns. Values extracted from the operation are
subject to the same IR invalidation rules after a rewrite.

Use the borrowed rewriter only inside the body, and perform every mutation
through `PatternRewriter` or `RewriterBase` so MLIR's greedy driver observes it.
Each native pattern owns a BEAM callback process, which serializes its lifecycle
and state transitions. This does not make borrowed MLIR handles transferable
between processes or callback invocations.
