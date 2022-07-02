defmodule PDLTest do
  use ExUnit.Case
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

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
  test "AreEqualOp" do
    ctx = MLIR.Context.create()
    CAPI.mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create(ctx, @apply_rewrite_op_patterns)
    ir_module = MLIR.Module.create(ctx, @apply_rewrite_op_ir)
    MLIR.Operation.verify!(pattern_module)
    MLIR.Operation.verify!(ir_module)
    pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
    pattern_set = CAPI.beaverRewritePatternSetGet(ctx)
    pattern_set = CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pdl_pattern)
    region = CAPI.mlirOperationGetFirstRegion(ir_module)
    result = CAPI.beaverApplyOwnedPatternSet(region, pattern_set)

    assert result
           |> Exotic.Value.fetch(MLIR.CAPI.MlirLogicalResult, :value)
           |> Exotic.Value.extract() != 0

    ir_string = MLIR.Operation.to_string(ir_module)
    assert not String.contains?(ir_string, "test.op")
    assert String.contains?(ir_string, "test.success")
    CAPI.mlirContextDestroy(ctx)
  end

  # TODO: figure out why custom asm format of pdl doesn't work
  @are_equal_op_pdl Path.join(__DIR__, "pdl_erase_and_create.mlir") |> File.read!()

  test "AreEqualOp pdl version" do
    ctx = MLIR.Context.create()
    CAPI.mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create(ctx, @are_equal_op_pdl)
    assert not MLIR.Module.is_null(pattern_module), "fail to parse module"
    ir_module = MLIR.Module.create(ctx, @apply_rewrite_op_ir)
    MLIR.Operation.verify!(pattern_module)
    MLIR.Operation.verify!(ir_module)
    pattern_string = MLIR.Operation.to_string(pattern_module)
    assert String.contains?(pattern_string, "test.op")
    assert String.contains?(pattern_string, "test.success2")
    pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
    pattern_set = CAPI.beaverRewritePatternSetGet(ctx)
    pattern_set = CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pdl_pattern)
    region = CAPI.mlirOperationGetFirstRegion(ir_module)
    result = CAPI.beaverApplyOwnedPatternSet(region, pattern_set)

    assert MLIR.LogicalResult.success?(result), "fail to apply pattern"

    ir_string = MLIR.Operation.to_string(ir_module)
    assert not String.contains?(ir_string, "test.op")
    assert String.contains?(ir_string, "test.success2")
    CAPI.mlirContextDestroy(ctx)
  end

  test "replace tosa" do
    defmodule TestTOSAPatterns do
      alias Beaver.MLIR

      pattern replace_add_op(_t = %TOSA.Add{operands: [a, b], results: [res], attributes: []}) do
        %TOSA.Sub{operands: [a, b]}
      end
    end

    ctx = MLIR.Context.create(allow_unregistered: true)

    ir_module =
      mlir do
        module do
          Func.func test_multi_broadcast(
                      function_type:
                        Attribute.type(
                          Type.function(
                            [
                              Type.ranked_tensor([1, 3], Type.f32()),
                              Type.ranked_tensor([2, 1], Type.f32())
                            ],
                            [Type.ranked_tensor([2, 3], Type.f32())]
                          )
                        )
                    ) do
            region do
              block entry(
                      arg0 :: Type.ranked_tensor([1, 3], Type.f32()),
                      arg1 :: Type.ranked_tensor([2, 1], Type.f32())
                    ) do
                v0 = TOSA.add(arg0, arg1) >>> Type.ranked_tensor([2, 3], Type.f32())
                Func.return(v0)
              end
            end
          end
        end
      end

    MLIR.Operation.verify!(ir_module)

    pattern_set =
      MLIR.PatternSet.get()
      |> MLIR.PatternSet.insert(TestTOSAPatterns.replace_add_op())

    MLIR.PatternSet.apply!(ir_module, pattern_set)
    MLIR.Operation.verify!(ir_module, dump_if_fail: true)
    ir_module = ir_module |> MLIR.Transforms.canonicalize() |> MLIR.Pass.Composer.run!()
    ir_string = MLIR.Operation.to_string(ir_module)
    assert not String.contains?(ir_string, "tosa.add"), ir_string
    assert String.contains?(ir_string, "tosa.sub"), ir_string
  end
end
