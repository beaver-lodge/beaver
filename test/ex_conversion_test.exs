defmodule ExConversionTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Ex
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex.MaterializeBoundVariables
  alias Beaver.MLIR.Dialect.Ex.ExpandCase
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

  @control_flow_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 1 : i64} : () -> i64
      %1 = "ex.lit"() {value = 2 : i64} : () -> i64
      %2 = "ex.cmp"(%0, %1) {predicate = "slt"} : (i64, i64) -> i64
      %3 = "ex.if"(%2) ({
        "ex.yield"(%0) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      }, {
        "ex.yield"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      }) {operandSegmentSizes = array<i32: 1>} : (i64) -> i64
      "ex.return"(%3) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
    }) {sym_name = "main"} : () -> ()
  }
  """

  @case_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 1 : i64} : () -> i64
      %1 = "ex.case"(%0) ({
      ^bb0:
        "ex.clause"() {patterns = array<i64: 1>} : () -> ()
        %2 = "ex.lit"() {value = 10 : i64} : () -> i64
        "ex.yield"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      ^bb1:
        "ex.clause"() {patterns = array<i64: 2>} : () -> ()
        %3 = "ex.lit"() {value = 20 : i64} : () -> i64
        "ex.yield"(%3) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      ^bb2:
        "ex.clause"() {patterns = array<i64>} : () -> ()
        %4 = "ex.lit"() {value = 30 : i64} : () -> i64
        "ex.yield"(%4) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      }) {operandSegmentSizes = array<i32: 1>} : (i64) -> i64
      "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
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

  test "lowers ex control flow ops to scf", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@control_flow_module, ctx: ctx)
      |> MLIR.verify!()

    converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "arith.cmpi"
    assert rendered =~ "scf.if"
    assert rendered =~ "scf.yield"
  end

  test "rejects unknown ex.cmp predicates", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.lit"() {value = 2 : i64} : () -> i64
            %2 = "ex.cmp"(%0, %1) {predicate = "bogus"} : (i64, i64) -> i64
            "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    assert {:error, %MLIR.Conversion.Error{} = error} = Plan.run(Ex.plan(), module)
    assert Exception.message(error) =~ "ex.cmp"
  end

  test "expands ex.case into nested ex.if and lowers to scf", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@case_module, ctx: ctx)
      |> MLIR.verify!()

    module = ExpandCase.run!(module)
    expanded = MLIR.to_string(module, generic: true)
    refute expanded =~ ~s{"ex.case"}
    refute expanded =~ ~s{"ex.clause"}
    assert expanded =~ ~s{"ex.if"}
    assert expanded =~ ~s{"ex.cmp"}

    converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "arith.cmpi"
    assert rendered =~ "scf.if"
    assert rendered =~ "scf.yield"
  end

  test "rejects a case without a catch-all clause", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.case"(%0) ({
            ^bb0:
              "ex.clause"() {patterns = array<i64: 1>} : () -> ()
              %2 = "ex.lit"() {value = 10 : i64} : () -> i64
              "ex.yield"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
            }) {operandSegmentSizes = array<i32: 1>} : (i64) -> i64
            "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    assert_raise ArgumentError, ~r/without a catch-all/, fn ->
      ExpandCase.run!(module)
    end
  end
end
