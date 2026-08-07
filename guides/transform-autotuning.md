# Transform schedule execution and autotuning

`Beaver.MLIR.Transform` can execute an upstream named Transform dialect
sequence directly against a payload. The schedule may be textual MLIR, MLIR
bytecode, a `Beaver.MLIR.Module`, a nested `Beaver.MLIR.Operation`, or a
`Beaver.MLIR.Transform.Schedule.Resolved` value.

```elixir
alias Beaver.MLIR

{:ok, result} =
  MLIR.Transform.apply_named_sequence(payload, transform_ir,
    sequence: "__transform_main",
    expensive_checks: true,
    enforce_single_top_level_transform_op: true
  )

result.payload
result.diagnostics
```

Both checks default to `true`. Diagnostics are returned as portable trees whose
locations are strings, so errors remain printable after a temporary schedule
context is destroyed. Failures are tagged as invalid schedules, propagated
silenceable failures, or definite failures. Transform handles remain internal
to upstream `TransformState`; consumed or invalidated handles never become
long-lived Elixir resources.

## A tiling and vectorization search space

Tune operations leave choices in Transform IR instead of hiding them in host
language control flow. This example selects a tile size, tiles a matched
`linalg.matmul`, and chooses whether to vectorize its enclosing isolated op.

```elixir
defmodule MatmulSchedule do
  use Beaver.MLIR.Transform.Schedule.DSL

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Transform

  defschedule tiling_and_vectorization do
    sequence "__transform_main", [root >>> any_op()] do
      tile = knob("tile_size", [8, 16, 32], type: param(MLIR.Type.i64()))

      operation_names =
        MLIR.Attribute.array(
          [MLIR.Attribute.string("linalg.matmul")],
          ctx: Beaver.Env.context()
        )

      matmul = Transform.structured_match(target: root, ops: operation_names) >>> any_op()

      static_sizes =
        MLIR.Attribute.dense_array([-1], Beaver.Native.I64, ctx: Beaver.Env.context())

      scalable_sizes =
        MLIR.Attribute.dense_array([false], Beaver.Native.Bool, ctx: Beaver.Env.context())

      [tiled, _loop] =
        Transform.structured_tile_using_for(
          target: matmul,
          dynamic_sizes: [tile],
          static_sizes: static_sizes,
          scalable_sizes: scalable_sizes
        ) >>> [any_op(), any_op()]

      alternatives "vectorize" do
        branch do
          isolated = MLIR.Attribute.unit(ctx: Beaver.Env.context())

          function =
            Transform.get_parent_op(target: tiled, isolated_from_above: isolated) >>> any_op()

          _vectorized =
            Transform.structured_vectorize_children_and_apply_patterns(target: function) >>>
              any_op()
        end

        branch do
          :ok
        end
      end
    end
  end
end

schedule_context = Beaver.MLIR.Context.create()
transform_ir = MatmulSchedule.tiling_and_vectorization(ctx: schedule_context)
```

`defschedule` emits and verifies an ordinary `Beaver.MLIR.Module`; it does not
retain a parallel Elixir choice graph. Each declaration also carries its
Elixir file and line as an MLIR location. The caller owns `transform_ir` and
`schedule_context` and must destroy the module before the context after the
search is complete.

### Applying patterns with CSE

`transform.apply_patterns` runs a greedy pattern driver on the body of its
target. The upstream operation carries an `apply_cse` unit attribute: when
set, common subexpression elimination is interleaved with pattern application
until a fixpoint, so duplicated subexpressions are folded into a single
definition. The DSL passes the upstream spelling through unchanged.

```elixir
defschedule canonicalize_with_cse do
  sequence "__transform_main", [root >>> any_op()] do
    apply_cse = MLIR.Attribute.unit(ctx: Beaver.Env.context())

    Transform.apply_patterns(target: root, apply_cse: apply_cse) do
      region do
        block do
          Transform.apply_patterns_canonicalization()
        end
      end
    end >>> []
  end
end
```

Omitting the `apply_cse` attribute runs the same patterns without the CSE
interleaving. Both variants are ordinary Transform IR, so they round-trip
through text and bytecode and resolve and execute like any other schedule.

Discovery and enumeration are deterministic. An alternatives choice is visited
before choices in its regions, and choices from unselected regions do not
inflate the candidate set.

```elixir
alias Beaver.MLIR.Transform.Schedule

{:ok, analysis} = Schedule.analyze(transform_ir)
{:ok, candidates} = Schedule.enumerate(transform_ir, max_candidates: 1_000)

# candidates starts with:
# %{"tile_size" => 8, "vectorize" => 0}
# %{"tile_size" => 8, "vectorize" => 1}
```

An explicit map is the simplest resolver. Functions and modules implementing
`Beaver.MLIR.Transform.Resolver` are also accepted.

```elixir
resolved =
  Schedule.resolve!(transform_ir, %{
    "tile_size" => 16,
    "vectorize" => 0
  })

File.write!("schedule.mlirbc", Schedule.serialize(resolved, :bytecode))

# Replay does not call the resolver or enumerate candidates again.
MLIR.Transform.apply_named_sequence!(payload, resolved)
```

Resolution writes `selected` and `selected_region` back into a copied schedule,
then records text, bytecode, the active choice map, and a SHA-256 digest. Knobs
in inactive alternatives may remain visibly unresolved because the replayed
schedule cannot enter those regions.

## Bounded concurrent evaluation

`Beaver.MLIR.Transform.Tuner` evaluates candidates with `Task.async_stream/3`
using bounded concurrency and ordered result collection. The evaluation hook
owns the scoring system; Beaver only records values, metadata, failures,
timeouts, and cancellation deterministically.

```elixir
alias Beaver.MLIR.Transform.Tuner

cancellation = Tuner.Cancellation.new()

result =
  Tuner.search!(transform_ir, fn resolved, candidate ->
    artifact =
      MLIR.CompilationRuntime.compile!(payload_source,
        transform_schedule: resolved,
        cache: {:file, ".beaver/cache"},
        target: target
      )

    # Replace this hook with application-specific compile/run benchmarking.
    {:ok, benchmark.(artifact), %{choices: candidate.choices}}
  end,
    max_concurrency: 4,
    timeout: 30_000,
    cancellation: cancellation
  )

winner =
  Tuner.select(result, fn successful ->
    Enum.min_by(successful, & &1.value)
  end)

File.write!("winner.mlirbc", winner.schedule.bytecode)
```

`CompilationRuntime` includes the resolved schedule identity in its cache key.
Changing a selection therefore invalidates the transformed artifact, while an
identical resolved schedule reuses it. A running evaluator can inspect
`candidate.cancelled?`; `Tuner.Cancellation.cancel/1` also prevents queued work
from starting. Non-cooperative work is bounded by the per-candidate timeout.

### Correlating candidates with MLIR actions

The tuner emits search and candidate `:start`/`:stop` events below
`[:beaver, :mlir, :compilation, :autotuning, ...]`. Candidate stop metadata
includes the resolved schedule digest and status. The evaluator context also
contains `telemetry_metadata`; pass it to `MLIR.ActionTracing` to tag every
lower-level pass, rewrite, tiling, and vectorization action with the candidate
that caused it:

```elixir
Tuner.search!(transform_ir, fn resolved, candidate ->
  ctx = MLIR.Context.create()
  tracing =
    MLIR.ActionTracing.attach(ctx, metadata: candidate.telemetry_metadata)

  try do
    artifact =
      MLIR.CompilationRuntime.compile!(payload_source,
        context: ctx,
        transform_schedule: resolved,
        cache: {:file, ".beaver/cache"},
        target: target
      )

    MLIR.ActionTracing.drain(tracing)
    {:ok, benchmark.(artifact)}
  after
    MLIR.ActionTracing.detach(tracing)
    MLIR.Context.destroy(ctx)
  end
end)
```

Each concurrent candidate should own its context and tracing session. This
keeps native action observation context-scoped while the shared telemetry sink
can group events by `candidate_index`, `choices`, and
`transform_schedule_digest`.

## Constraints without a bundled solver

`Schedule.constraints/2` exports every reachable
`transform.smt.constrain_params` as generic MLIR text plus its operand/result
counts and alternatives guards. This works without an SMT installation.

```elixir
{:ok, constraints} = Schedule.constraints(transform_ir)

solver = fn constraints, selections ->
  MySolver.validate(constraints, selections)
end

{:ok, resolved} = Schedule.resolve(transform_ir, choices, solver: solver)
resolved.solver_metadata
```

For reusable adapters, implement `Beaver.MLIR.Transform.Solver`. Beaver does not
assume that one score, solver, or hardware benchmark is authoritative; the
adapter boundary keeps those policies outside the reproducible schedule.
