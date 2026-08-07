# Shadow Wavefront: audit, tune, receipt, replay

The Shadow Wavefront experiment loop treats a compilation decision as a
recorded, replayable experiment instead of a one-off optimization. The first
workload is Triton's `ttg.convert_layout` placement: layout conversions mark
points where data may move between encodings, and choosing between layout
strategies is exactly the kind of decision that should produce evidence.

The loop has four stages:

```text
input IR → layout audit → Transform candidates → receipt → replay
```

## 1. Audit the input

With Triton dialects registered, `Beaver.MLIR.Triton.LayoutAudit` collects
every `ttg.convert_layout` in a module together with its source/target layout
encodings, tensor facts, and source location:

```elixir
context = Beaver.MLIR.Context.create(all_dialects: false)
Beaver.Triton.register(context)

module = Beaver.MLIR.Module.create!(File.read!("ttgir.mlir"), ctx: context)
audit = Beaver.MLIR.Triton.LayoutAudit.audit(module)

audit.operation_count
audit.convert_layouts
```

The audit is a report, not a cost model. Unknown encodings are preserved as
raw MLIR text instead of being silently dropped.

## 2. Author candidates

Layout strategies are expressed as Transform schedule alternatives using the
schedule DSL, so each candidate is ordinary upstream Transform IR:

```elixir
defmodule LayoutSchedule do
  use Beaver.MLIR.Transform.Schedule.DSL

  defschedule layout_strategy do
    sequence "__transform_main", [root >>> any_op()] do
      alternatives "strategy" do
        branch do
          # strategy A: keep layouts, no conversion
        end

        branch do
          # strategy B: convert to shared layout
        end
      end
    end
  end
end
```

## 3. Run the closed loop

`Beaver.Shadow.Runner` enumerates candidates deterministically,
evaluates each one, and records one receipt per candidate:

```elixir
alias Beaver.Shadow.Runner

run =
  Runner.run(ttgir_text, LayoutSchedule.layout_strategy(ctx: context),
    evaluator: fn resolved, candidate ->
      # application-specific scoring; metadata flows into the receipt
      {:ok, score, %{artifact: ..., trace: ...}}
    end
  )

run.receipts
run.winner
```

The default evaluator is a deterministic surrogate (score by schedule digest
and choices), so the loop runs on CPU without a GPU or a compilation cache.
Pass a custom evaluator to use `CompilationRuntime` cache hits, `ActionTracing`
evidence, or real kernel latency.

### GPU evaluator

When a CUDA driver is present, `Beaver.Shadow.GPU` replaces the
surrogate with a real hardware loop: it packages the payload into `gpu.binary`
(NVVM or ROCDL), loads it through the Zig CUDA runner, launches a kernel, and
records device facts plus native-time measurements in the same receipt schema:

```elixir
alias Beaver.Shadow.GPU

run = GPU.run(ttgir_text, LayoutSchedule.layout_strategy(ctx: context))

run.winner.user_metadata.device
run.winner.user_metadata.durations
```

On machines without `libcuda` the GPU evaluator degrades to a recorded
`:cuda_unavailable` failure instead of crashing, so the same experiment loop
keeps running on CPU-only machines.

Pass a `CompilationCache` to reuse the packaged `gpu.binary` across candidates
and repeated experiments. The cache key is `(source, target)` only — candidate
choices and durations never invalidate it — so the first candidate pays the
`gpu-module-to-binary` compilation and every later candidate or rerun reports a
cache hit:

```elixir
cache = start_supervised!({Beaver.MLIR.CompilationCache.Memory, []})

run = GPU.run(ttgir_text, LayoutSchedule.layout_strategy(ctx: context),
  cache: {:memory, cache}
)

run.winner.artifact.cache
```

## 4. Serialize and replay

Every receipt is versioned and JSON-serializable. Durations and other
observations never enter `Receipt.identity/1`, so identical experiments
compare equal regardless of when they ran:

```elixir
json = run.winner |> Receipt.encode!()
receipt = json |> Receipt.decode!()

assert Receipt.identity(receipt) == Receipt.identity(run.winner)

{:ok, result} =
  Runner.replay(receipt, payload_module,
    expensive_checks: false
  )
```

Replay uses the winner's resolved schedule bytecode directly and does not call
the resolver or enumerate candidates again.

## Current boundary

This stage runs compiler-side only. Layout cost is an explicit, replaceable
heuristic — it does not claim to equal real GPU performance. Real kernel
latency arrives with the Zig CUDA runner, and the Triton plugin path will let
the same loop drive passes inside an actual Triton pipeline.

## Real workload trial

`Beaver.Shadow.OptimizationTrial` is the first consumer that compares two real
pipeline variants on a real Triton kernel instead of a synthetic fixture. It
uses the upstream Triton matmul TTGIR (`test/fixtures/triton/ttgir_matmul.mlir`)
and answers one question: how many `ttg.convert_layout` operations survive?

- baseline: `convert-triton-to-tritongpu` only — 24 conversions;
- optimized: plus `tritongpu-remove-layout-conversions` — 5 conversions;
- reduction: 19 conversions eliminated (reproducible on every run).

This is the first reproducible, non-trivial optimization evidence produced by
the Shadow Wavefront loop: the same input, two real pipeline strategies, and a
structural audit that records the difference without human IR reading.

The same kernel also runs end to end on an NVIDIA GPU: `Beaver.Triton.compile_to_llvm/2`
runs the full NVIDIA pipeline (shared-memory allocation, warp-group allocation,
warp-specialize lowering, NVGPU lowering) before translating to PTX and loading
through `Beaver.MLIR.CUDA`. On an RTX 5070 the trial reported ~149μs baseline
vs ~160μs optimized (speedup ≈ 0.93) for the 64x64 f32 matmul — a useful
probe, but single-kernel latency is noisy, so treat small speedup deltas as
measurement noise rather than conclusions.

## External measurement harness

Since M3, the loop has a second role: measuring Triton's layout decisions the
way upstream reviewers need — structurally, reproducibly, and without reading
IR by hand. Three pieces support that:

- `Beaver.Shadow.Corpus` pins the fixture set and its structural facts
  (`ttgir_matmul.mlir`, `ttir_attention.mlir`, `ttgir_convert_layout.mlir`).
  The attention fixture is a flash-attention-style TTIR (online-softmax
  accumulators as `scf.for` iter_args + `tt.dot`), and the remat fixture is a
  TTGIR backward slice that mirrors the triton-lang/triton#11026 pattern.
- `mix shadow.probe <fixture>` lowers a fixture in a child BEAM and reports
  `:ok`, `:error`, or `:crash` with the offending pass. Native crashes in the
  prebuilt (which would kill the caller) become deterministic failure receipts.
  On the pinned prebuilt, the dot-free remat slice crashes
  `tritongpu-accelerate-matmul` (exit 139) — a regression probe that flips to
  `:ok` once the prebuilt is fixed or bumped.
- `mix shadow.report` measures the whole corpus and prints a Markdown table
  (baseline/optimized `ttg.convert_layout` counts, lowering capability, probe
  status, LLVM revision) that can be pasted into upstream issues or PRs.
