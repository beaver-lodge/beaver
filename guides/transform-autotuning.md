# Transform execution

Beaver executes native MLIR Transform dialect sequences. The experimental
Elixir schedule DSL, candidate enumeration/resolution, tuner, and solver
adapters have been retired; see the [core API migration](core-api-migration.md).
Applications that search for a schedule now own selection and scoring.

## Apply a named sequence

`Beaver.MLIR.Transform.apply_named_sequence/3` accepts a module, an operation,
textual MLIR, or bytecode. It copies the schedule into the payload context and
releases the temporary schedule after execution. The payload remains owned by
the caller. `execute/3` is an alias.

```elixir
alias Beaver.MLIR

schedule = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
"""

# payload is an MLIR.Module or Operation owned by the caller.
{:ok, result} = MLIR.Transform.execute(payload, schedule)
transformed_payload = result.payload
```

The default sequence is `__transform_main`. Set `sequence: "my_sequence"` to
select another symbol. Expensive handle checks and single-top-level-transform
checks default to enabled. Errors retain processed diagnostics and distinguish
invalid input, silenceable execution failures, and definite failures.

## Replay selected Tune operations

Upstream Tune operations remain available. Their choices must be selected in
the input IR before execution. For example:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %tile = transform.tune.knob<"tile"> = 16 from options = [8, 16] -> !transform.any_param
    transform.tune.alternatives<"route"> selected_region = 1 {
      transform.yield
    }, {
      transform.yield
    }
    transform.yield
  }
}
```

Serialize a parsed schedule with `MLIR.Bytecode.write!/1`, then pass those bytes
to `Transform.execute/3`. The tagged forms `{:text, text}` and
`{:bytecode, bytes}` are also accepted. No resolver or search runs implicitly.

Use `Beaver.Walker`, operation attributes, and the native Transform dialect
builders when an application needs to inspect or construct schedule IR.
`Beaver.MLIR.ActionTracing` and conversion profiling remain available for
measurement. Solver integration and application cache policy belong to the
consumer.
