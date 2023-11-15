defmodule PDLTest do
  use Beaver.Case
  use Beaver
  alias Beaver.MLIR
  alias MLIR.Type
  alias MLIR.CAPI
  alias MLIR.Dialect.{Func, TOSA}
  import MLIR.Transforms
  require Func
  require TOSA
  require Type

  @moduletag :pdl

  @apply_rewrite_op_patterns """
  module @patterns {
    pdl_interp.func @matcher(%root : !pdl.operation) {
      %test_attr = pdl_interp.create_attribute unit
      %attr = pdl_interp.get_attribute "test_attr" of %root
      pdl_interp.are_equal %test_attr, %attr : !pdl.attribute -> ^pat, ^end

    ^pat:
      pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

    ^end:
      pdl_interp.finalize
    }

    module @rewriters {
      pdl_interp.func @success(%root : !pdl.operation) {
        %op = pdl_interp.create_operation "test.success1"
        pdl_interp.erase %root
        pdl_interp.finalize
      }
    }
  }
  """

  @apply_rewrite_op_ir """
  module @ir attributes { test.are_equal_1 } {
    "test.op"() { test_attr } : () -> ()
  }
  """

  test "AreEqualOp", test_context do
    ctx = test_context[:ctx]
    CAPI.mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create(ctx, @apply_rewrite_op_patterns)

    inspector = fn
      {:successor, %MLIR.Block{} = successor}, acc ->
        {{:successor, successor}, acc}

      {:argument, %MLIR.Value{}} = argument, acc ->
        {argument, acc}

      {:result, %MLIR.Value{}} = result, acc ->
        {result, acc}

      {:operand, %MLIR.Value{}} = operand, acc ->
        {operand, acc}

      {name, %MLIR.Attribute{} = attribute}, acc ->
        {{name, attribute}, acc}

      %MLIR.Operation{} = op, acc ->
        {op, [MLIR.Operation.name(op) | acc]}

      %element{} = mlir, acc ->
        {mlir, [element | acc]}
    end

    {mlir, acc} =
      pattern_module
      |> Beaver.Walker.traverse([], inspector, inspector)

    assert acc == [
             "builtin.module",
             MLIR.Region,
             MLIR.Block,
             "builtin.module",
             MLIR.Region,
             MLIR.Block,
             "pdl_interp.func",
             MLIR.Region,
             MLIR.Block,
             "pdl_interp.finalize",
             "pdl_interp.finalize",
             "pdl_interp.erase",
             "pdl_interp.erase",
             "pdl_interp.create_operation",
             "pdl_interp.create_operation",
             MLIR.Block,
             MLIR.Region,
             "pdl_interp.func",
             MLIR.Block,
             MLIR.Region,
             "builtin.module",
             "pdl_interp.func",
             MLIR.Region,
             MLIR.Block,
             "pdl_interp.finalize",
             "pdl_interp.finalize",
             MLIR.Block,
             MLIR.Block,
             "pdl_interp.record_match",
             "pdl_interp.record_match",
             MLIR.Block,
             MLIR.Block,
             "pdl_interp.are_equal",
             "pdl_interp.are_equal",
             "pdl_interp.get_attribute",
             "pdl_interp.get_attribute",
             "pdl_interp.create_attribute",
             "pdl_interp.create_attribute",
             MLIR.Block,
             MLIR.Region,
             "pdl_interp.func",
             MLIR.Block,
             MLIR.Region,
             "builtin.module"
           ]

    assert mlir
           |> MLIR.CAPI.mlirOperationEqual(MLIR.Operation.from_module(pattern_module))
           |> Beaver.Native.to_term()

    ir_module = MLIR.Module.create(ctx, @apply_rewrite_op_ir)
    MLIR.Operation.verify!(pattern_module)
    MLIR.Operation.verify!(ir_module)
    pattern_set = CAPI.beaverRewritePatternSetGet(ctx)
    pattern_set = CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pattern_module)

    region =
      ir_module
      |> MLIR.Operation.from_module()
      |> CAPI.mlirOperationGetFirstRegion()

    result = CAPI.beaverApplyOwnedPatternSetOnRegion(region, pattern_set)

    assert MLIR.LogicalResult.success?(result)

    ir_string = MLIR.to_string(ir_module)
    assert not String.contains?(ir_string, "test.op")
    assert String.contains?(ir_string, "test.success")
  end

  @are_equal_op_pdl Path.join(__DIR__, "pdl_erase_and_create.mlir") |> File.read!()

  test "AreEqualOp pdl version" do
    ctx = MLIR.Context.create()
    CAPI.mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create(ctx, @are_equal_op_pdl)
    assert not MLIR.Module.is_null(pattern_module), "fail to parse module"
    ir_module = MLIR.Module.create(ctx, @apply_rewrite_op_ir)
    MLIR.Operation.verify!(pattern_module)
    MLIR.Operation.verify!(ir_module)
    pattern_string = MLIR.to_string(pattern_module)
    assert String.contains?(pattern_string, "test.op")
    assert String.contains?(pattern_string, "test.success2")
    pattern_set = CAPI.beaverRewritePatternSetGet(ctx)
    pattern_set = CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pattern_module)
    region = ir_module |> MLIR.Operation.from_module() |> CAPI.mlirOperationGetFirstRegion()
    result = CAPI.beaverApplyOwnedPatternSetOnRegion(region, pattern_set)

    assert MLIR.LogicalResult.success?(result), "fail to apply pattern"

    ir_string = MLIR.to_string(ir_module)
    assert not String.contains?(ir_string, "test.op")
    assert String.contains?(ir_string, "test.success2")
    CAPI.mlirContextDestroy(ctx)
  end

  test "replace tosa", test_context do
    defmodule TestTOSAPatterns do
      def gen_ir_module(ctx) do
        mlir ctx: ctx do
          module do
            Func.func test_multi_broadcast(
                        function_type:
                          Type.function(
                            [
                              Type.ranked_tensor([1, 3], Type.f32()),
                              Type.ranked_tensor([2, 1], Type.f32())
                            ],
                            [Type.ranked_tensor([2, 3], Type.f32())]
                          )
                      ) do
              region do
                block entry(
                        a >>> Type.ranked_tensor([1, 3], Type.f32()),
                        b >>> Type.ranked_tensor([2, 1], Type.f32())
                      ) do
                  res =
                    TOSA.add(a, b, one: MLIR.Attribute.integer(MLIR.Type.i32(), 1)) >>>
                      Type.ranked_tensor([2, 3], Type.f32())

                  res1 = TOSA.add(res, b) >>> Type.ranked_tensor([2, 3], Type.f32())
                  Func.return(res1) >>> []
                end
              end
            end
          end
        end
      end

      import Beaver.Pattern

      defpat replace_add_op(benefit: 10) do
        a = value()
        b = value()
        res_t = type()
        {op, _} = TOSA.add(a, b) >>> {:op, [res_t]}

        rewrite op do
          r = TOSA.sub(a, b) >>> res_t
          replace(op, with: r)
        end
      end

      defpat replace_multi_add_op() do
        one = Attribute.integer(MLIR.Type.i32(), 1)
        _x = %Range{first: 1, last: 10, step: 2}
        ty = Type.ranked_tensor([2, 3], Type.f32())
        a = value()
        b = value()
        res = TOSA.add(a, b, one: one) >>> ty
        {op, _t} = TOSA.add(res, b) >>> {:op, ty}

        rewrite op do
          types = [Type.ranked_tensor([2, 3], Type.f32())]
          a = TOSA.sub(a, b) >>> types
          a = TOSA.sub(a, b) >>> types
          a = TOSA.sub(a, b) >>> types
          a = TOSA.sub(a, b, one: one) >>> types
          {r, _} = TOSA.sub(a, b) >>> {:op, ty}
          replace(op, with: r)
        end
      end

      defpat replace_multi_add_op1() do
        one = Attribute.integer(MLIR.Type.i32(), 1)
        ty = Type.ranked_tensor([2, 3], Type.f32())
        a = value()
        b = value()
        res = TOSA.add(a, b, one: one) >>> ty
        {op, _t} = TOSA.add(res, b) >>> {:op, ty}

        rewrite op do
          {r, _} = TOSA.sub(a, b) >>> {:op, ty}
          replace(op, with: r)
        end
      end

      defpat replace_multi_add_op2() do
        one = Attribute.integer(MLIR.Type.i32(), 1)
        types = [Type.ranked_tensor([2, 3], Type.f32())]
        a = value()
        b = value()

        res = TOSA.add(a, b, one: one) >>> types
        ty = type()
        {op, _t} = TOSA.add(res, b) >>> {:op, ty}

        rewrite op do
          {r, _} = TOSA.sub(a, b) >>> {:op, ty}
          replace(op, with: r)
        end
      end

      defpat replace_multi_add_op3() do
        one = Attribute.integer(MLIR.Type.i32(), 1)
        types = [Type.ranked_tensor([2, 3], Type.f32())]
        a = value()
        b = value()
        res = TOSA.add(a, b, one: one) >>> types
        {op, _t} = TOSA.add(res, b) >>> {:op, types}

        rewrite op do
          r = TOSA.sub(a, b) >>> types
          replace(op, with: r)
        end
      end
    end

    ctx = test_context[:ctx]
    opts = [ctx: ctx]

    for pattern <- [
          TestTOSAPatterns.replace_add_op(),
          TestTOSAPatterns.replace_multi_add_op(opts),
          TestTOSAPatterns.replace_multi_add_op1(opts),
          TestTOSAPatterns.replace_multi_add_op2(opts),
          TestTOSAPatterns.replace_multi_add_op3(opts)
        ] do
      ir_module = TestTOSAPatterns.gen_ir_module(ctx)
      MLIR.Operation.verify!(ir_module)
      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "tosa.sub"), ir_string

      MLIR.Pattern.apply!(ir_module, [pattern])
      |> MLIR.Operation.verify!(debug: true)
      |> MLIR.Transforms.canonicalize()
      |> MLIR.Pass.Composer.run!()

      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "tosa.add"), ir_string
      assert String.contains?(ir_string, "tosa.sub"), ir_string
    end
  end

  test "toy compiler with pass", test_context do
    ctx = test_context[:ctx]

    defmodule ToyPass do
      @moduledoc false
      use MLIR.Pass, on: "func.func"
      import Beaver.Pattern

      defpat replace_add_op() do
        a = value()
        b = value()
        res = type()
        {op, _t} = TOSA.add(a, b) >>> {:op, [res]}

        rewrite op do
          {r, _} = TOSA.sub(a, b) >>> {:op, [res]}
          replace(op, with: r)
        end
      end

      def run(%MLIR.Operation{} = operation) do
        with "func.func" <- MLIR.Operation.name(operation),
             attributes <- Beaver.Walker.attributes(operation),
             2 <- Enum.count(attributes),
             {:ok, _} <- MLIR.Pattern.apply_(operation, [replace_add_op(benefit: 2)]) do
          :ok
        end
      end
    end

    ir =
      ~m"""
      module {
        func.func @tosa_add(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
          %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
          return %0 : tensor<2x3xf32>
        }
      }
      """.(ctx)
      |> MLIR.Pass.Composer.nested("func.func", [
        ToyPass
      ])
      |> canonicalize
      |> MLIR.Pass.Composer.run!()

    ir_string = MLIR.to_string(ir)
    assert not (ir_string =~ "tosa.add")
    assert ir_string =~ "tosa.sub"
  end
end
