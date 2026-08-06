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

Keeping attachment out of schema construction makes the IRDL module safe to
inspect, serialize, and test without mutating dialect registration. It also
makes failures attributable to either schema verification or interface
attachment instead of interleaving the two phases.

Slang assigns every generated IRDL operation the source location of its
declaration. Invalid nested constraints therefore report the `deftype`,
`defattr`, or `defop` line rather than an implementation line inside Slang.
