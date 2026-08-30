# Lowering a Slang dialect

Beaver exposes MLIR's dialect conversion driver without requiring a native
pass. `Beaver.MLIR.Conversion.Plan` provides a declarative, inspectable, and
scoped composition layer over conversion targets, type converters, materializations,
and conversion patterns. The conversion pipeline is declared in Elixir, while
the conversion driver runs on the context's native worker pool.

This example defines a small Slang operation and lowers it to the `arith`
dialect using a `Conversion.Plan`. The resulting IR can then use MLIR's standard
`arith`-to-LLVM and `func`-to-LLVM passes.

## Ex provider boundary

Beaver owns the `ex` dialect schema and the provider-neutral external
compiler-kernel ABI. Evolving production lowering belongs to Batata and is
loaded through `Conversion.Plan.add_external_pattern_population/3`; Beaver
does not locate or download Batata artifacts.

`Beaver.MLIR.Conversion.Ex.Stage0.manifest/0` describes the only Ex-specific
C++ implementation retained in Beaver. It is a frozen, versioned bootstrap
seed for Batata's restricted compiler-kernel source and a differential oracle,
not a production provider. Runtime and standard-library patterns remain in the
BEAM reference plan so Stage 0 can close a clean bootstrap without extending
the C++ seed. A production compiler must select a verified native provider and
fail before conversion if that artifact cannot be loaded; it must never fall
back implicitly to Stage 0 or the BEAM callbacks.

```elixir
defmodule Counter do
  use Beaver.Slang, name: "counter"

  defop inc(value = Beaver.MLIR.Type.i32()),
    do: [Beaver.MLIR.Type.i32()]
end

alias Beaver.MLIR

ctx = MLIR.Context.create()
Beaver.Slang.load(ctx, Counter)

module =
  MLIR.Module.create!(
    ~S"""
    module {
      func.func @increment(%arg0: i32) -> i32 {
        %0 = "counter.inc"(%arg0) : (i32) -> i32
        return %0 : i32
      }
    }
    """,
    ctx: ctx
  )

# Define a declarative, reusable conversion plan
plan =
  MLIR.Conversion.Plan.new(
    mode: :full,
    folding_mode: :after_patterns,
    build_materializations: true
  )
  |> MLIR.Conversion.Plan.add_legal_dialect("builtin")
  |> MLIR.Conversion.Plan.add_legal_dialect("func")
  |> MLIR.Conversion.Plan.add_legal_dialect("arith")
  |> MLIR.Conversion.Plan.add_illegal_dialect("counter")
  |> MLIR.Conversion.Plan.add_conversion(fn type -> type end, version: "1.0")
  |> MLIR.Conversion.Plan.add_conversion_pattern(
    "counter.inc",
    fn operation, [input], rewriter ->
      context = MLIR.context(operation)
      base = MLIR.ConversionPatternRewriter.as_base(rewriter)
      type = MLIR.Value.type(input)
      location = MLIR.Operation.location(operation)

      one =
        %Beaver.Changeset{name: "arith.constant", context: context, location: location}
        |> Beaver.Changeset.add_argument(value: MLIR.Attribute.integer(type, 1))
        |> Beaver.Changeset.add_result(type)
        |> MLIR.Operation.create()

      add =
        %Beaver.Changeset{name: "arith.addi", context: context, location: location}
        |> Beaver.Changeset.add_argument([input, MLIR.Operation.result(one, 0)])
        |> Beaver.Changeset.add_result(type)
        |> MLIR.Operation.create()

      MLIR.RewriterBase.set_insertion_point_before(base, operation)
      MLIR.RewriterBase.insert(base, one)
      MLIR.RewriterBase.insert(base, add)
      MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, add)
      :ok
    end,
    version: "1.0"
  )

# Execute the plan. Native target, converter, and pattern set are allocated
# and cleaned up automatically for the duration of the run.
{:ok, ^module, _diagnostics} = MLIR.Conversion.Plan.run(plan, module)

# The custom dialect is gone. Continue with upstream conversion passes.
pass_manager = MLIR.CAPI.mlirPassManagerCreate(ctx)

MLIR.CAPI.mlirPassManagerAddOwnedPass(
  pass_manager,
  MLIR.CAPI.mlirCreateConversionArithToLLVMConversionPass()
)

MLIR.CAPI.mlirPassManagerAddOwnedPass(
  pass_manager,
  MLIR.CAPI.mlirCreateConversionConvertFuncToLLVMPass()
)

{:ok, _pass_diagnostics} = MLIR.PassManager.run(pass_manager, module)
MLIR.PassManager.destroy(pass_manager)
```

## Plan Inspection and Callback Versioning

`MLIR.Conversion.Plan.declaration/1` returns a deterministic map of plan metadata
with function closures and runtime state omitted:

```elixir
decl = MLIR.Conversion.Plan.declaration(plan)
# => %{
#   mode: :full,
#   timeout: 30000,
#   folding_mode: :after_patterns,
#   build_materializations: true,
#   entries: [
#     %{kind: :add_legal_dialect, dialect: "builtin"},
#     ...,
#     %{kind: :add_conversion, version: "1.0"},
#     %{kind: :add_conversion_pattern, root: "counter.inc", benefit: 1, one_to_n: false, timeout: nil, version: "1.0"}
#   ]
# }
```

Callbacks registered with plan builders accept optional `:version` metadata.
When `:version` is omitted, the callback is visibly marked `:unversioned`.
Declaration metadata is only deterministic and reproducible across runs or processes
when callback versions are explicitly specified.

## Resource Ownership, Borrowing, and Timeout

`Conversion.Plan` structs do not hold native resources; a single plan can be
safely reused across multiple fresh `MLIR.Context` instances.

When `Plan.run/2` or `Plan.run!/2` executes:

1. Fresh native `MLIR.ConversionTarget`, `MLIR.TypeConverter`, and `MLIR.RewritePatternSet` objects are created.
2. Target legality rules, conversions, materializations, and patterns are populated in declaration order.
3. Conversion runs with the specified `:timeout` (default 30,000 ms).
4. The mutable pattern set is transferred to `Conversion.apply/5`; its native worker releases the frozen set even if the caller terminates. On a normal or error return, the remaining resources are then released in reverse dependency order: pattern set, type converter, conversion target.

## Dynamic legality

Static legality applies to every instance of an operation or dialect. For an
instance-dependent decision, register a callback:

```elixir
plan =
  MLIR.Conversion.Plan.add_dynamically_legal_op(
    plan,
    "counter.inc",
    fn operation ->
      if safe_to_keep?(operation), do: :legal, else: :no_opinion
    end,
    version: "1.0"
  )
```

Callbacks may return `:legal`, `:illegal`, or `:no_opinion`. Exceptions and
`{:error, reason}` results are preserved in `Beaver.MLIR.Conversion.Error`
rather than being flattened into a generic MLIR failure.

## Native rewrite descriptors

A plan can include descriptors produced by `Beaver.Pattern.Native.defrewrite/3`:

```elixir
plan =
  MLIR.Conversion.Plan.add_pattern(
    plan,
    MyNativePatterns.lower_counter(),
    version: "1.0"
  )
```

This composes with the existing callback-backed Native DSL; `Conversion.Plan`
does not introduce another matching language. Use `add_conversion_pattern/4`
when a rewrite needs type-converted operand adaptors, and `add_pattern/3` for an
ordinary Native descriptor. Declaration metadata records the descriptor name,
root, benefit, and explicit version, but omits its callbacks and runtime state.

## 1:N conversion and materialization

Use `Plan.add_1_to_n_conversion/3` to return zero, one, or several
target types. A conversion pattern created with `one_to_n: true` receives one
list of converted values per original operand and can replace each result with
a value range through
`ConversionPatternRewriter.replace_op_with_multiple/3`.

Source, target, and 1:N target materializations are registered on the plan with
`Plan.add_source_materialization/3`, `Plan.add_target_materialization/3`, and
`Plan.add_1_to_n_target_materialization/3`. Their rewriter, values, types, and
locations are scoped to the callback. Do not retain those handles after the
callback returns.

## Low-Level Escape Hatch

For advanced scenarios requiring manual resource lifecycle management, low-level
APIs (`MLIR.ConversionTarget`, `MLIR.TypeConverter`, `MLIR.RewritePatternSet`, and
`MLIR.Conversion.apply/5`) remain fully supported:

```elixir
target = MLIR.ConversionTarget.create(ctx)
converter = MLIR.TypeConverter.create(conversion: fn type -> type end)
patterns = MLIR.RewritePatternSet.create(ctx)
MLIR.ConversionPattern.add(patterns, "counter.inc", converter, callback, ctx: ctx)

try do
  {:ok, ^module, _diagnostics} = MLIR.Conversion.full(module, target, patterns)
after
  MLIR.TypeConverter.destroy(converter)
  MLIR.ConversionTarget.destroy(target)
end
```
