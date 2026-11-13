defmodule PDLTest do
  use Beaver.Case, async: true
  import MLIR.CAPI
  import MLIR.Transform
  @moduletag :pdl
  @apply_rewrite_op_patterns "module @patterns {\n  pdl_interp.func @matcher(%root : !pdl.operation) {\n    %test_attr = pdl_interp.create_attribute unit\n    %attr = pdl_interp.get_attribute \"test_attr\" of %root\n    pdl_interp.are_equal %test_attr, %attr : !pdl.attribute -> ^pat, ^end\n\n  ^pat:\n    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end\n\n  ^end:\n    pdl_interp.finalize\n  }\n\n  module @rewriters {\n    pdl_interp.func @success(%root : !pdl.operation) {\n      %op = pdl_interp.create_operation \"test.success1\"\n      pdl_interp.erase %root\n      pdl_interp.finalize\n    }\n  }\n}\n"
  @apply_rewrite_op_ir "module @ir attributes { test.are_equal_1 } {\n  \"test.op\"() { test_attr } : () -> ()\n}\n"
  def apply_patterns(pattern_module, ir_module, cb) do
    MLIR.verify!(pattern_module)
    MLIR.verify!(ir_module)
    pdl_pat_mod = mlirPDLPatternModuleFromModule(pattern_module)
    cfg = mlirGreedyRewriteDriverConfigCreate()
    frozen_pat_set = pdl_pat_mod |> mlirRewritePatternSetFromPDLPatternModule() |> mlirFreezeRewritePattern()

    try do
      result = mlirApplyPatternsAndFoldGreedily(ir_module, frozen_pat_set, cfg)
      assert MLIR.LogicalResult.success?(result)
      cb.(ir_module)
    after
      mlirGreedyRewriteDriverConfigDestroy(cfg)
      mlirPDLPatternModuleDestroy(pdl_pat_mod)
      MLIR.FrozenRewritePatternSet.destroy(frozen_pat_set)
    end
  end

  test "AreEqualOp", %{ctx: ctx} do
    mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create!(@apply_rewrite_op_patterns, ctx: ctx)

    inspector = fn
      {:successor, %MLIR.Block{} = successor}, acc -> {{:successor, successor}, acc}
      {:argument, %MLIR.Value{}} = argument, acc -> {argument, acc}
      {:result, %MLIR.Value{}} = result, acc -> {result, acc}
      {:operand, %MLIR.Value{}} = operand, acc -> {operand, acc}
      {name, %MLIR.Attribute{} = attribute}, acc -> {{name, attribute}, acc}
      %MLIR.Operation{} = op, acc -> {op, [MLIR.Operation.name(op) | acc]}
      %element{} = mlir, acc -> {mlir, [element | acc]}
    end

    {mlir, acc} = pattern_module |> Beaver.Walker.traverse([], inspector, inspector)
    expected = ~w(
      builtin.module Region Block builtin.module Region Block pdl_interp.func Region Block
      pdl_interp.finalize pdl_interp.finalize pdl_interp.erase pdl_interp.erase
      pdl_interp.create_operation pdl_interp.create_operation Block Region pdl_interp.func Block Region
      builtin.module pdl_interp.func Region Block pdl_interp.finalize pdl_interp.finalize Block Block
      pdl_interp.record_match pdl_interp.record_match Block Block pdl_interp.are_equal
      pdl_interp.are_equal pdl_interp.get_attribute pdl_interp.get_attribute pdl_interp.create_attribute
      pdl_interp.create_attribute Block Region pdl_interp.func Block Region builtin.module
    )
    expected = Enum.map(expected, fn name -> %{"Region" => MLIR.Region, "Block" => MLIR.Block}[name] || name end)
    assert acc == expected
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
      MLIR.apply!(ir_module, [pattern]) |> MLIR.verify!() |> MLIR.Transform.canonicalize() |> Beaver.Composer.run!()
      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "tosa.add"), ir_string
      assert String.contains?(ir_string, "tosa.sub"), ir_string
    end
  end

  for p <- [ToyPass, ToyPassWithInit] do
    test "toy compiler with pass: #{p}", %{ctx: ctx} do
      ir =
        sigil_m(
          "module {\n  func.func @tosa_add(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {\n    %0 = \"tosa.add\"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>\n    return %0 : tensor<2x3xf32>\n  }\n}\n",
          []
        )
        |> Beaver.Deferred.resolve(ctx)
        |> Beaver.Composer.append(unquote(p))
        |> canonicalize
        |> Beaver.Composer.run!()

      ir_string = MLIR.to_string(ir)
      assert not (ir_string =~ "tosa.add")
      assert ir_string =~ "tosa.sub"
    end
  end
end
