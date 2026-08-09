# Defining dynamic dialects with Slang

`Beaver.Slang` defines MLIR types, attributes, and operations in Elixir, then
builds an IRDL schema that MLIR can verify and load at runtime. The declaration
names are preserved in the schema, so diagnostics and generated IR remain
readable.

## A complete dialect

```elixir
defmodule Geometry do
  use Beaver.Slang, name: "geometry"

  defconstraint integer_like do
    all_of([base("!builtin.integer"), any()])
  end

  defconstraint direction_value do
    any_of([
      Beaver.MLIR.Attribute.string("left"),
      Beaver.MLIR.Attribute.string("right")
    ])
  end

  deftype index(element = ^integer_like)
  deftype token()
  defattr direction(value = ^direction_value)

  defop consume_token(value = base(token()))

  defop sequence(head = any(), tail = variadic(any()), fallback = optional(any())),
    results: [value: optional(any())]

  defop scope(),
    attributes: [label: base("#builtin.string")],
    regions: [body: {:region, args: [], size: 1}],
    traits: [:isolated_from_above, :no_terminator]

  defop yield(), traits: [:terminator]
end
```

Build and inspect the schema without changing the context's registered dynamic
dialects:

```elixir
schema = Geometry.__slang_dialect__(ctx)
Beaver.MLIR.verify!(schema)
IO.puts(Beaver.MLIR.to_string(schema, generic: true))
```

Load it when operations, types, or attributes from the dialect are needed:

```elixir
result = Beaver.Slang.load(ctx, Geometry)
true = Beaver.MLIR.LogicalResult.success?(result)
```

## Constraints

`any()` accepts any type or attribute. `is(value)` requires one exact MLIR type
or attribute. `any_of(list)` and `all_of(list)` compose constraints, including
constraints already produced by another combinator.

`base("!dialect.type")` and `base("#dialect.attribute")` constrain the base type
or attribute without constraining its parameters. A Slang type or attribute
constructor can also supply the base reference, as in `base(token())` above.

Use `defconstraint` for a reusable constraint and reference it with `^name`.
`defalias` remains available as an equivalent concise spelling.

## Names and cardinality

Names are inferred from assignments and variables:

```elixir
deftype pair(left = any(), right = any())
defop add(lhs = any(), rhs = any()), results: [sum: any()]
```

For expressions that do not bind a variable, pass `parameter_names`,
`operand_names`, or `result_names` explicitly:

```elixir
deftype pair(any(), any()), parameter_names: [:left, :right]

defop add(any(), any()),
  operand_names: [:lhs, :rhs],
  results: [any()],
  result_names: [:sum]
```

Operation attributes and regions are keyword lists, so their keys are their
schema names. `optional` and `variadic` apply to operation operands and results,
which are the entities for which upstream IRDL exposes variadicity. Type and
attribute parameters, operation attributes, and regions are single entries.

Regions accept these descriptors:

- `:any` allows any region shape.
- `{:sized, positive_integer}` constrains the block count.
- `{:region, args: constraints, size: positive_integer}` constrains entry-block
  arguments, block count, or both. Omit either option to leave it unconstrained.

## Schema and runtime interfaces

The IRDL module and external operation interfaces have different lifetimes.
`__slang_dialect__/1` only constructs the schema. `Beaver.Slang.load/2`
canonicalizes and verifies it, registers the dialect, and only then attaches
the requested built-in dynamic traits:

- `:terminator`
- `:isolated_from_above`
- `:no_terminator`

Custom dynamic traits attach Elixir verification callbacks under a stable
identity. The same identity shares one MLIR TypeID across every operation in a
context:

```elixir
defmodule MyDialect.ValidOperands do
end

def verify_operands(operation) do
  operand_count =
    Beaver.MLIR.CAPI.mlirOperationGetNumOperands(operation)
    |> Beaver.Native.to_term()

  if operand_count > 0,
    do: :ok,
    else: {:error, :missing_operand}
end

defop checked(value = optional(any())),
  traits: [
    {MyDialect.ValidOperands,
     verify: &__MODULE__.verify_operands/1}
  ]
```

Use `:verify` for ordinary invariants and `:verify_regions` for invariants that
need access to nested regions. A verifier accepts one borrowed operation and
returns `:ok`, `true`, `{:ok, value}`, `:error`, `false`, or
`{:error, reason}`. Failures, exceptions, unavailable callback owners, and
timeouts become MLIR diagnostics and verification failures.

Each operation attachment has a dedicated callback process. Custom trait
TypeIDs and callback ownership belong to the MLIR context and are released by
`Beaver.MLIR.Context.destroy/1`. Context multithreading must remain enabled so
parse and verification work can run outside BEAM scheduler threads. Direct
`Beaver.MLIR.Trait.attach_custom/5` calls accept a `:timeout` option, which
defaults to 30 seconds.

The operation passed to a verifier is borrowed and valid only until the
callback returns. Do not send it to another process, store it, or use it after
the callback.

Keeping attachment out of schema construction makes the IRDL module safe to
inspect, serialize, and test without mutating dialect registration. It also
makes failures attributable to either schema verification or interface
attachment instead of interleaving the two phases.

### External operation interfaces

The `interfaces:` option attaches MLIR fallback models implemented by Elixir
callbacks. This lets dynamic operations participate in analyses, transforms,
and rewrites without a C++ or TableGen definition:

```elixir
defmodule Effects do
  use Beaver.Slang, name: "effects"

  def pure(_operation), do: :pure
  def speculatable(_operation), do: :speculatable

  def forward_apply(operation, _rewriter, _results, state) do
    handle = Beaver.MLIR.CAPI.mlirOperationGetOperand(operation, 0)
    payload = Beaver.MLIR.TransformOpInterface.payload_ops(state, handle)
    {:ok, %{0 => {:ops, payload}}}
  end

  def forward_effects(operation, effects) do
    operand = Beaver.MLIR.CAPI.mlirOperationGetOpOperand(operation, 0)
    result = Beaver.MLIR.Operation.result(operation, 0)
    Beaver.MLIR.MemoryEffects.only_reads_handle(effects, operand)
    Beaver.MLIR.MemoryEffects.produces_handle(effects, result)
    Beaver.MLIR.MemoryEffects.only_reads_payload(effects)
  end

  defop constant(),
    interfaces: [
      memory_effects: &__MODULE__.pure/1,
      conditionally_speculatable: &__MODULE__.speculatable/1
    ]

  defop forward(handle = base("!transform.any_op")),
    do: [base("!transform.any_op")],
    interfaces: [
      memory_effects: &__MODULE__.forward_effects/2,
      transform_op: [apply: &__MODULE__.forward_apply/4]
    ]
end
```

The supported keys are:

- `:memory_effects` accepts a callback of arity one that returns `:pure` or a
  list of `Beaver.MLIR.MemoryEffects` specifications. An arity-two callback may
  instead add transform handle effects to its borrowed effects list.
- `:conditionally_speculatable` returns `:not_speculatable`, `:speculatable`,
  or `:recursively_speculatable`.
- `:transform_op` requires an `:apply` callback of arity four and accepts an
  optional `:allows_repeated_handle_operands` callback. Apply callbacks map op
  results with `{:ops, values}`, `{:values, values}`, or `{:params, values}`.
- `:pattern_descriptor` requires a `:populate_patterns` callback of arity two
  and accepts a state-aware callback of arity three. Add patterns through
  `Beaver.MLIR.RewritePatternSet`.

Each attachment has a dedicated BEAM callback process and belongs to one MLIR
context. Native callers wait outside normal BEAM schedulers, callback waits are
bounded to 30 seconds by default, and `Beaver.MLIR.Context.destroy/1` releases
the model and its callback process. Direct `attach` functions accept a
`:timeout` option.

Operations, effect lists, transform rewriters, transform results and states,
and rewrite-pattern sets delivered to callbacks are borrowed. They are valid
only until that callback returns: do not send them to another process, store
them, or use them later. A callback may use the ordinary Beaver APIs during
that interval, but it must not synchronously re-enter the same interface
attachment. Context multithreading must remain enabled for callback-backed
interfaces.

Callback exceptions are emitted as MLIR diagnostics at the operation location
and logged on the BEAM side. The native fallback is conservative: failed
memory-effect callbacks add an unknown write, failed speculation callbacks are
not speculatable, and failed transform callbacks return a definite failure.

Dynamic dialect and trait registration is local to an `MLIR.Context`; it is not
stored in IRDL or operation bytecode. Call `Beaver.Slang.load/2` once for every
new context before parsing text or reading bytecode that uses the dialect.

Slang assigns every generated IRDL operation the source location of its
declaration. Invalid nested constraints therefore report the `deftype`,
`defattr`, or `defop` line rather than an implementation line inside Slang.
