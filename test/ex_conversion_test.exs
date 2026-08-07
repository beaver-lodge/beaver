defmodule ExConversionTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Ex
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex.MaterializeBoundVariables
  alias Beaver.MLIR.Dialect.Ex, as: ExDialect

  @moduletag :smoke

  @scalar_module ~S"""
  module {
    func.func @add(%a: i64, %b: i64) -> i64 {
      %r = arith.addi %a, %b : i64
      return %r : i64
    }
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 1 : i64} : () -> i64
      %1 = "ex.lit"() {value = 2 : i64} : () -> i64
      %2 = "ex.add"(%0, %1) : (i64, i64) -> i64
      %3 = "ex.call"(%0, %1) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 2>} : (i64, i64) -> !ex.dyn
      "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
    }) {sym_name = "main"} : () -> ()
  }
  """

  @bind_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 42 : i64} : () -> i64
      %1 = "ex.var"() {name = "x"} : () -> !ex.unbound
      %2 = "ex.bind"(%1, %0) : (!ex.unbound, i64) -> !ex.bound
      "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (!ex.bound) -> ()
    }) {sym_name = "main"} : () -> ()
  }
  """

  test "lowers the ex scalar subset to func/arith and llvm", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@scalar_module, ctx: ctx)
      |> MLIR.verify!()

    plan = Ex.plan()
    assert {:ok, ^module, _diagnostics} = Plan.run(plan, module)

    rendered = MLIR.to_string(module, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "arith.addi"
    assert rendered =~ "func.call"
    assert rendered =~ "func.return"

    pass_manager = MLIR.CAPI.mlirPassManagerCreate(ctx)

    MLIR.CAPI.mlirPassManagerAddOwnedPass(
      pass_manager,
      MLIR.CAPI.mlirCreateConversionArithToLLVMConversionPass()
    )

    MLIR.CAPI.mlirPassManagerAddOwnedPass(
      pass_manager,
      MLIR.CAPI.mlirCreateConversionConvertFuncToLLVMPass()
    )

    assert {:ok, _} = MLIR.PassManager.run(pass_manager, module)
    MLIR.PassManager.destroy(pass_manager)

    rendered_llvm = MLIR.to_string(module)
    assert rendered_llvm =~ "llvm.func"
    assert rendered_llvm =~ "llvm.add"
  end

  test "materializes ex.var/ex.bind into SSA", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@bind_module, ctx: ctx)
      |> MLIR.verify!()
      |> MaterializeBoundVariables.run!()

    assert converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex.var"
    refute rendered =~ "ex.bind"
    assert rendered =~ "arith.constant"
    assert rendered =~ "func.return"
  end

  test "fails explicitly on a bare ex.var", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.var"() {name = "x"} : () -> !ex.unbound
            "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    module = MaterializeBoundVariables.run!(module)

    assert {:error, %MLIR.Conversion.Error{}} = Plan.run(Ex.plan(), module)
  end

  test "converts ex term types feeding ex.return", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.call"(%0, %0) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 2>} : (i64, i64) -> !ex.dyn
            "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.dyn) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "func.call"
    assert rendered =~ "func.return"
  end

  test "lowers sub/mul and function arguments", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0(%arg0: i64, %arg1: i64):
            %0 = "ex.mul"(%arg0, %arg1) : (i64, i64) -> i64
            %1 = "ex.sub"(%0, %arg0) : (i64, i64) -> i64
            "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ ~s{function_type = (i64, i64) -> i64, sym_name = "main"}
    assert rendered =~ "^bb0(%arg0: i64, %arg1: i64)"
    assert rendered =~ "arith.muli"
    assert rendered =~ "arith.subi"
  end
end
