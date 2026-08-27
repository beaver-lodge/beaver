# Profiling dialect conversion

`Beaver.MLIR.Conversion.profile/5` and
`Beaver.MLIR.Conversion.Plan.profile/2` return the normal conversion result
plus a bounded, machine-readable receipt. Profiling is opt-in; ordinary
conversion does not walk the IR inventory or aggregate callback timings.

```elixir
{converted, receipt} =
  Beaver.MLIR.Conversion.Plan.profile!(
    Beaver.MLIR.Conversion.Ex.plan(),
    module
  )
```

The receipt reports four upper-level cost signals:

* native conversion duration and an unattributed residual,
* conversion-target lock wait,
* BEAM callback service time by callback kind, and
* estimated callback boundary wait by callback kind, including transport and
  queueing.

`callback_wait_sum_ns`, callback service, and boundary overhead are additive
work counters, not a partition of wall time. Their sums can exceed native
duration when MLIR blocks multiple native threads concurrently. Boundary wait
also spans native and BEAM platform clocks, so individual samples may be zero
or differ slightly from the enclosing native span at clock resolution.

`unattributed_residual_ns` clamps native duration minus lock wait and aggregate
callback wait at zero. It is a heuristic remainder, not a claimed native
compute bound. `hotspots` ranks that residual together with accumulated
callback and lock cost counters in deterministic order; it is evidence
attribution, not an exact wall-time partition.

The callback list retains one aggregate per callback kind rather than one item
per invocation. The current BEAM receive loop services callbacks serially, so
its service `max_in_flight` is either zero or one; this does not claim that only
one native caller can be waiting. BEAM process memory is sampled at the stage
boundaries and every 256 callbacks to bound observer cost.

The IR inventory records module, function, and operation counts before and
after conversion. Pair those fields with the caller's semantic fingerprint and
failure frontier when comparing two compiler versions. Beaver intentionally
does not infer whether two converted programs are semantically equivalent.
The receipt also reports process CPU time and operating-system peak RSS before,
after, and as a stage delta. Platforms without a portable peak-RSS source
report zero rather than substituting a different memory definition.

## Reproducible scaling probe

The repository includes a synthetic Ex conversion probe:

```console
mix run profile/ex_conversion_profile.exs -- --functions 64 --iterations 7 --warmup 2
```

It alternates unprofiled and profiled conversions of fresh, identical modules,
checks their generic-IR SHA-256 fingerprints, and emits JSON. Observer overhead
is the median of paired ratios, which keeps runner-speed drift from being
mistaken for instrumentation cost. The output includes the final bounded
receipt. Run it at several function counts to distinguish fixed bridge cost
from IR scaling.

For a whole compiler pipeline, place `Plan.profile/2` around the Ex conversion
stage and use `Beaver.MLIR.ActionTracing` around the later MLIR passes. The two
receipts have different roles: conversion profiling measures the synchronous
native/BEAM callback bridge, while action tracing separates MLIR pass actions.
