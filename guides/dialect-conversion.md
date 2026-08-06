# Lowering a Slang dialect

Beaver exposes MLIR's dialect conversion driver without requiring a native
pass. The conversion target, type converter, and patterns below are all built
in Elixir; the driver still runs on the context's native worker pool.

This example defines a small Slang operation and lowers it to the `arith`
dialect. The resulting IR can then use MLIR's standard `arith`-to-LLVM and
`func`-to-LLVM passes.

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

target =
  MLIR.ConversionTarget.create(ctx)
  |> MLIR.ConversionTarget.add_legal_dialect("builtin")
  |> MLIR.ConversionTarget.add_legal_dialect("func")
  |> MLIR.ConversionTarget.add_legal_dialect("arith")
  |> MLIR.ConversionTarget.add_illegal_dialect("counter")

# Identity conversion keeps i32 unchanged. A callback may instead return a
# list of types through add_1_to_n_conversion/2.
converter = MLIR.TypeConverter.create(conversion: fn type -> type end)
patterns = MLIR.RewritePatternSet.create(ctx)

MLIR.ConversionPattern.add(
  patterns,
  "counter.inc",
  converter,
  fn operation, [input], rewriter ->
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    type = MLIR.Value.type(input)
    location = MLIR.Operation.location(operation)

    one =
      %Beaver.Changeset{name: "arith.constant", context: ctx, location: location}
      |> Beaver.Changeset.add_argument(value: MLIR.Attribute.integer(type, 1))
      |> Beaver.Changeset.add_result(type)
      |> MLIR.Operation.create()

    add =
      %Beaver.Changeset{name: "arith.addi", context: ctx, location: location}
      |> Beaver.Changeset.add_argument([input, MLIR.Operation.result(one, 0)])
      |> Beaver.Changeset.add_result(type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, one)
    MLIR.RewriterBase.insert(base, add)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, add)
    :ok
  end,
  ctx: ctx
)

try do
  {:ok, ^module, diagnostics} =
    MLIR.Conversion.full(module, target, patterns,
      folding_mode: :after_patterns,
      build_materializations: true
    )

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
after
  # Destroy the frozen/pattern set before its referenced converter. Passing a
  # mutable set to Conversion.full/4 does this automatically.
  MLIR.TypeConverter.destroy(converter)
  MLIR.ConversionTarget.destroy(target)
end
```

## Dynamic legality

Static legality applies to every instance of an operation or dialect. For an
instance-dependent decision, register a callback:

```elixir
target =
  MLIR.ConversionTarget.add_dynamically_legal_op(
    target,
    "counter.inc",
    fn operation ->
      if safe_to_keep?(operation), do: :legal, else: :no_opinion
    end
  )
```

Callbacks may return `:legal`, `:illegal`, or `:no_opinion`. Exceptions and
`{:error, reason}` results are preserved in `Beaver.MLIR.Conversion.Error`
rather than being flattened into a generic MLIR failure.

## 1:N conversion and materialization

Use `TypeConverter.add_1_to_n_conversion/2` to return zero, one, or several
target types. A conversion pattern created with `one_to_n: true` receives one
list of converted values per original operand and can replace each result with
a value range through
`ConversionPatternRewriter.replace_op_with_multiple/3`.

Source, target, and 1:N target materializations are registered with
`add_source_materialization/2`, `add_target_materialization/2`, and
`add_1_to_n_target_materialization/2`. Their rewriter, values, types, and
locations are scoped to the callback. Do not retain those handles after the
callback returns.

Targets and type converters are explicit native owners. The native conversion
worker cleans up its `ConversionConfig` and any mutable pattern set frozen by
`Conversion.apply/5`, even if the calling process terminates. Standalone
targets and converters should use `ConversionTarget.with/3`,
`TypeConverter.with/2`, or an equivalent `try/after` as shown above.
