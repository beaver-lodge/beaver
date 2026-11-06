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

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Arith
  alias Beaver.MLIR.Type

  def square_to_mul(_pattern, operation, rewriter, state) do
    [input] = Enum.to_list(Beaver.Walker.operands(operation))
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(operation)
    type = MLIR.Value.type(input)

    mlir ctx: ctx, ip: base do
      replacement = Arith.muli(input, input) >>> type
      MLIR.RewriterBase.replace_op(base, operation, [replacement])
    end

    {:ok, state}
  end

  def inc_to_add1(_pattern, operation, rewriter, state) do
    [input] = Enum.to_list(Beaver.Walker.operands(operation))
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(operation)
    type = MLIR.Value.type(input)

    mlir ctx: ctx, ip: base do
      one = Arith.constant(value: MLIR.Attribute.integer(type, 1)) >>> type
      replacement = Arith.addi(input, one) >>> type
      MLIR.RewriterBase.replace_op(base, operation, [replacement])
    end

    {:ok, state}
  end
end

defmodule DogfoodTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias MLIR.Conversion.Plan
  alias MLIR.DogfoodDialect
  alias MLIR.DogfoodNativePatterns

  @source """
  module {
    func.func @kernel(%arg0: i32, %arg1: i32) -> (i32, i32) {
      %0 = "dogfood.square"(%arg0) : (i32) -> i32
      %1 = "dogfood.inc"(%arg1) : (i32) -> i32
      return %0, %1 : i32, i32
    }
  }
  """

  test "dogfood: Slang -> native rewrite -> conversion plan -> transform schedule -> bytecode",
       %{
         ctx: ctx
       } do
    assert Beaver.Slang.load(ctx, DogfoodDialect) |> MLIR.LogicalResult.success?()

    module = MLIR.Module.create!(@source, ctx: ctx)

    # A callback rewrite precedes dialect conversion.
    MLIR.Rewrite.apply_patterns!(module, [
      {"dogfood.square", &DogfoodNativePatterns.square_to_mul/4}
    ])

    rewritten = MLIR.to_string(module)
    refute rewritten =~ "dogfood.square"
    assert rewritten =~ "arith.muli"

    # 2. #486: a scoped Conversion Plan lowers dogfood.inc, reusing the same
    #    native matcher as its conversion pattern frontend.
    plan =
      Plan.new(
        mode: :full,
        folding_mode: :after_patterns,
        build_materializations: true,
        timeout: 5_000
      )
      |> Plan.add_legal_dialect("builtin")
      |> Plan.add_legal_dialect("func")
      |> Plan.add_legal_dialect("arith")
      |> Plan.add_illegal_dialect("dogfood")
      |> Plan.add_conversion(fn type -> type end, version: "1.0")
      |> Plan.add_pattern_population(fn patterns, _converter ->
        MLIR.RewritePatternSet.add(patterns, "dogfood.inc", &DogfoodNativePatterns.inc_to_add1/4,
          ctx: MLIR.CAPI.mlirRewritePatternSetGetContext(patterns)
        )

        :ok
      end)

    assert {:ok, ^module, _diagnostics} = Plan.run(plan, module)

    lowered = MLIR.to_string(module)
    refute lowered =~ "dogfood.inc"
    assert lowered =~ "arith.addi"
    MLIR.verify!(module)

    # The same plan declaration is reusable in a fresh MLIR context.
    ctx2 = MLIR.Context.create()
    assert Beaver.Slang.load(ctx2, DogfoodDialect) |> MLIR.LogicalResult.success?()
    module2 = MLIR.Module.create!(@source, ctx: ctx2)

    MLIR.Rewrite.apply_patterns!(module2, [
      {"dogfood.square", &DogfoodNativePatterns.square_to_mul/4}
    ])

    assert {:ok, _module2, _diagnostics} = Plan.run(plan, module2)
    refute MLIR.to_string(module2) =~ "dogfood"
    MLIR.Context.destroy(ctx2)

    # Compose the retained native primitives directly, then serialize the result.
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

    schedule = MLIR.Module.create!(schedule, ctx: ctx)
    root = schedule |> MLIR.Operation.from_module() |> Beaver.Walker.regions() |> Enum.at(0)

    root =
      root |> Beaver.Walker.blocks() |> Enum.at(0) |> Beaver.Walker.operations() |> Enum.at(0)

    options = MLIR.CAPI.mlirTransformOptionsCreate()

    try do
      result =
        MLIR.CAPI.mlirTransformApplyNamedSequence(
          MLIR.Operation.from_module(module),
          root,
          MLIR.Operation.from_module(schedule),
          options
        )

      assert MLIR.LogicalResult.success?(result)
    after
      MLIR.CAPI.mlirTransformOptionsDestroy(options)
      MLIR.Module.destroy(schedule)
    end

    bytecode = MLIR.Bytecode.write!(module)
    reparsed = MLIR.Module.create!(bytecode, ctx: ctx)
    MLIR.verify!(reparsed)
    assert MLIR.to_string(reparsed) =~ "arith.muli"
    assert MLIR.to_string(reparsed) =~ "arith.addi"
    MLIR.Module.destroy(reparsed)
    MLIR.Module.destroy(module)
  end
end
