defmodule PDLTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias MLIR.Type
  import MLIR.CAPI
  alias MLIR.Dialect.{Func, TOSA}
  import MLIR.Transform
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

  def apply_patterns(pattern_module, ir_module, cb) do
    MLIR.verify!(pattern_module)
    MLIR.verify!(ir_module)
    pdl_pat_mod = mlirPDLPatternModuleFromModule(pattern_module)

    frozen_pat_set =
      pdl_pat_mod |> mlirRewritePatternSetFromPDLPatternModule() |> mlirFreezeRewritePattern()

    result = beaverModuleApplyPatternsAndFoldGreedily(ir_module, frozen_pat_set)
    assert MLIR.LogicalResult.success?(result)
    cb.(ir_module)
    mlirPDLPatternModuleDestroy(pdl_pat_mod)
    mlirFrozenRewritePatternSetDestroy(frozen_pat_set)
  end

  test "AreEqualOp", %{ctx: ctx} do
    mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create!(@apply_rewrite_op_patterns, ctx: ctx)

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

    assert MLIR.equal?(mlir, MLIR.Operation.from_module(pattern_module))

    ir_module = MLIR.Module.create!(@apply_rewrite_op_ir, ctx: ctx)

    apply_patterns(pattern_module, ir_module, fn ir_module ->
      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "test.op")
      assert String.contains?(ir_string, "test.success")
    end)

    MLIR.Module.destroy(pattern_module)
    MLIR.Module.destroy(ir_module)
  end

  @are_equal_op_pdl Path.join(__DIR__, "pdl_erase_and_create.mlir") |> File.read!()

  test "AreEqualOp pdl version", %{ctx: ctx} do
    mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create!(@are_equal_op_pdl, ctx: ctx)
    assert not MLIR.null?(pattern_module), "fail to parse module"
    ir_module = MLIR.Module.create!(@apply_rewrite_op_ir, ctx: ctx)
    pattern_string = MLIR.to_string(pattern_module)
    assert String.contains?(pattern_string, "test.op")
    assert String.contains?(pattern_string, "test.success2")

    apply_patterns(pattern_module, ir_module, fn ir_module ->
      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "test.op")
      assert String.contains?(ir_string, "test.success2")
    end)

    MLIR.Module.destroy(pattern_module)
    MLIR.Module.destroy(ir_module)
  end

  test "replace tosa", %{ctx: ctx} do
    for pattern <- [
          TestTOSAPatterns.replace_add_op(),
          TestTOSAPatterns.replace_multi_add_op(),
          TestTOSAPatterns.replace_multi_add_op1(),
          TestTOSAPatterns.replace_multi_add_op2(),
          TestTOSAPatterns.replace_multi_add_op3()
        ] do
      ir_module = TestTOSAPatterns.gen_ir_module(ctx)
      MLIR.verify!(ir_module)
      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "tosa.sub"), ir_string

      MLIR.apply!(ir_module, [pattern])
      |> MLIR.verify!()
      |> MLIR.Transform.canonicalize()
      |> Beaver.Composer.run!()

      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "tosa.add"), ir_string
      assert String.contains?(ir_string, "tosa.sub"), ir_string
    end
  end

  test "toy compiler with pass", %{ctx: ctx} do
    ir =
      ~m"""
      module {
        func.func @tosa_add(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
          %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
          return %0 : tensor<2x3xf32>
        }
      }
      """.(ctx)
      |> Beaver.Composer.append(ToyPass)
      |> canonicalize
      |> Beaver.Composer.run!()

    ir_string = MLIR.to_string(ir)
    assert not (ir_string =~ "tosa.add")
    assert ir_string =~ "tosa.sub"
  end
end
