# Action Tracing

MLIR dispatches compiler work — pass execution, greedy rewrite iterations,
tiling, and other transformations — through `mlir::Action`. Action Tracing
observes and optionally controls those dispatches at a granularity that
pass-level IR dumps cannot reach: you see the *decision* to run a pass or
pattern, not just the IR before and after it.

Beaver exposes this through `Beaver.MLIR.ActionTracing`: attach a
context-scoped session, run a pass pipeline, drain structured events, and
optionally skip or limit actions by tag.

## Comparison with existing debug output

`mlir-opt --debug` prints IR around pass execution and stops for human
interaction. It is useful interactively, but it serializes full IR text and is
not a data source: there is no structured record of which action ran, how deep
it was nested, or how long it took.

Beaver's pass instrumentation (`MLIR.Pass` timing, `--mlir-timing` style
output) aggregates wall time per pass. That answers *how long*, but not *what
was attempted*: an iteration that found no rewrites, a pattern that was tried
and failed, or a tiling decision that was made are all invisible.

Action Tracing sits between the two:

- Events carry `tag`, `depth`, `description`, the affected IR units (operation
  name + location, without full IR text), and a nanosecond timestamp.
- `start`/`stop` pairs are correlated on the BEAM side so `stop` carries a
  `duration` measurement.
- Filters select tags or source locations up front, so unobserved actions
  never leave the native side.
- Skip/limit controls change what executes, enabling deterministic bisection:
  skip the first N occurrences of a tag and observe whether the failure
  reproduces.

## Quick start

```elixir
ctx = MLIR.Context.create()

session =
  MLIR.ActionTracing.attach(ctx,
    tags: ["pass-execution"],
    skip: %{"pass-execution" => 1}
  )

module = MLIR.Module.create!("module {}", ctx: ctx)
module |> Beaver.Composer.append("canonicalize") |> Beaver.Composer.run!()

MLIR.ActionTracing.drain(session)
#=> [%{"phase" => "before", "tag" => "pass-execution", ...}, ...]

MLIR.ActionTracing.detach(session)
```

`drain/1` returns the decoded events and also emits them as telemetry:

- `[:beaver, :mlir, :compilation, :action, :start]`
- `[:beaver, :mlir, :compilation, :action, :stop]`

Pass `drain_interval_ms:` to have the session drain on a timer and emit
telemetry automatically.

## Filtering and control

- `:tags` — list of action tags to observe.
- `:locations` — list of source location substrings; an action is observed when
  one of its context IR units carries a matching location.
- `:skip` — map of tag to a skip count; the first N occurrences are *not
  executed*.
- `:limit` — map of tag to an execution limit; further occurrences are skipped
  once the limit is reached.

## Threading and teardown

Native observers run on MLIR worker threads and never call BEAM APIs. Events
are queued under a mutex and drained by the BEAM, so multithreaded pipelines
are safe by construction. The session is owned by the native context and is
detached automatically when the context is destroyed; `detach/1` is idempotent
and safe to call from any process.

## Limitations and future work

- Event pairing keys on `(tag, depth)`; concurrent actions at the same tag and
  depth from different threads may be paired out of order.
- Stepping and breakpoint-based control (`Step`, `Next`, `Finish`, `Rerun`) are
  not exposed yet; the observer path is intended to stabilize before those
  controls are added.
- The supplemental C ABI (`mlir-c/Beaver/ActionTracing.h`) is a private Beaver
  interface, not a stable public MLIR ABI.
