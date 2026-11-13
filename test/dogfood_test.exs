defmodule Beaver.MLIR.DogfoodDialect do
  @moduledoc false
  use Beaver.Slang, name: "dogfood"
  alias Beaver.MLIR.Type
  defop square(value = Type.i32()), do: [Type.i32()]
  defop inc(value = Type.i32()), do: [Type.i32()]
end

defmodule Beaver.MLIR.DogfoodNativePatterns do
  @moduledoc false
  use Beaver
  use Beaver.Pattern.Native
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Arith
  alias Beaver.MLIR.Type

  defrewrite square_to_mul(operation, rewriter, _state),
    root: "dogfood.square",
    operands: [input],
    results: [_result],
    attributes: [] do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(operation)
    type = MLIR.Value.type(input)

    mlir ctx: ctx, ip: base do
      replacement = Arith.muli(input, input) >>> type
      MLIR.RewriterBase.replace_op(base, operation, [replacement])
    end

    :ok
  end

  defrewrite inc_to_add1(operation, rewriter, _state),
    root: "dogfood.inc",
    operands: [input],
    results: [_result],
    attributes: [] do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(operation)
    type = MLIR.Value.type(input)

    mlir ctx: ctx, ip: base do
      one = Arith.constant(value: MLIR.Attribute.integer(type, 1)) >>> type
      replacement = Arith.addi(input, one) >>> type
      MLIR.RewriterBase.replace_op(base, operation, [replacement])
    end

    :ok
  end
end

defmodule DogfoodTest do
  use Beaver.Case, async: true
  alias MLIR.Conversion.Plan
  alias MLIR.DogfoodDialect
  alias MLIR.DogfoodNativePatterns

  @source "module {\n  func.func @kernel(%arg0: i32, %arg1: i32) -> (i32, i32) {\n    %0 = \"dogfood.square\"(%arg0) : (i32) -> i32\n    %1 = \"dogfood.inc\"(%arg1) : (i32) -> i32\n    return %0, %1 : i32, i32\n  }\n}\n"
  test "dogfood: Slang -> native rewrite -> conversion plan -> transform schedule -> bytecode",
       %{ctx: ctx} do
    assert Beaver.Slang.load(ctx, DogfoodDialect) |> MLIR.LogicalResult.success?()
    module = MLIR.Module.create!(@source, ctx: ctx)
    # 1. #485: native matcher rewrites dogfood.square without hand-written
    #    Walker unpacking and without generating wrapper source.
    MLIR.Rewrite.apply_patterns!(module, [DogfoodNativePatterns.square_to_mul()])
    rewritten = MLIR.to_string(module)
    refute rewritten =~ "dogfood.square"
    assert rewritten =~ "arith.muli"
    # 2. #486: a scoped Conversion Plan lowers dogfood.inc, reusing the same
    #    native matcher as its conversion pattern frontend.
    plan =
      Plan.new(mode: :full, folding_mode: :after_patterns, build_materializations: true, timeout: 5000)
      |> Plan.add_legal_dialects(~w(builtin func arith))
      |> Plan.add_illegal_dialect("dogfood")
      |> Plan.add_conversion(fn type -> type end, version: "1.0")
      |> Plan.add_pattern(DogfoodNativePatterns.inc_to_add1())

    assert {:ok, ^module, _diagnostics} = Plan.run(plan, module)
    lowered = MLIR.to_string(module)
    refute lowered =~ "dogfood.inc"
    assert lowered =~ "arith.addi"
    MLIR.verify!(module)
    # The same plan declaration is reusable in a fresh MLIR context.
    ctx2 = MLIR.Context.create()
    assert Beaver.Slang.load(ctx2, DogfoodDialect) |> MLIR.LogicalResult.success?()
    module2 = MLIR.Module.create!(@source, ctx: ctx2)
    MLIR.Rewrite.apply_patterns!(module2, [DogfoodNativePatterns.square_to_mul()])
    assert {:ok, _module2, _diagnostics} = Plan.run(plan, module2)
    refute MLIR.to_string(module2) =~ "dogfood"
    MLIR.Context.destroy(ctx2)
    # Compose the retained native primitives directly, then serialize the result.
    schedule =
      "module attributes {transform.with_named_sequence} {\n  transform.named_sequence @__transform_main(%root: !transform.any_op) {\n    transform.apply_patterns to %root {\n      transform.apply_patterns.canonicalization\n    } : !transform.any_op\n    transform.yield\n  }\n}\n"

    assert {:ok, _} = MLIR.Transform.apply_named_sequence(module, schedule)
    bytecode = MLIR.Bytecode.write!(module)
    reparsed = MLIR.Module.create!(bytecode, ctx: ctx)
    MLIR.verify!(reparsed)
    assert MLIR.to_string(reparsed) =~ "arith.muli"
    assert MLIR.to_string(reparsed) =~ "arith.addi"
    MLIR.Module.destroy(reparsed)
    MLIR.Module.destroy(module)
  end
end
