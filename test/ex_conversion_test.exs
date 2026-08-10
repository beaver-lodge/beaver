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
      %3 = "ex.call"(%0, %1) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i64, i64) -> !ex.dyn
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

  @multi_pattern_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 1 : i64} : () -> i64
      %1 = "ex.case"(%0) ({
      ^bb0:
        "ex.clause"() {patterns = array<i64: 1, 2>} : () -> ()
        %2 = "ex.lit"() {value = 10 : i64} : () -> i64
        "ex.yield"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      ^bb1:
        "ex.clause"() {patterns = array<i64>} : () -> ()
        %3 = "ex.lit"() {value = 30 : i64} : () -> i64
        "ex.yield"(%3) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      }) {operandSegmentSizes = array<i32: 1>} : (i64) -> i64
      "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
    }) {sym_name = "main"} : () -> ()
  }
  """

  @guard_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 1 : i64} : () -> i64
      %1 = "ex.lit"() {value = 1 : i64} : () -> i64
      %2 = "ex.case"(%0) ({
      ^bb0:
        "ex.clause"(%1) {patterns = array<i64: 1>} : (i64) -> ()
        %3 = "ex.lit"() {value = 10 : i64} : () -> i64
        "ex.yield"(%3) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      ^bb1:
        "ex.clause"() {patterns = array<i64>} : () -> ()
        %4 = "ex.lit"() {value = 30 : i64} : () -> i64
        "ex.yield"(%4) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      }) {operandSegmentSizes = array<i32: 1>} : (i64) -> i64
      "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
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
            %1 = "ex.call"(%0, %0) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i64, i64) -> !ex.dyn
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

  test "allows scalar-typed ex.call results", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.call"(%0) {callee = "id", arity = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>} : (i64) -> i64
            "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "func.call"
    refute rendered =~ "!ex.dyn"
  end

  test "allows heterogeneous-typed ex.call arguments", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.dyn
            %2 = "ex.call"(%1, %0) {callee = "f", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (!ex.dyn, i64) -> i64
            "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)
    assert MLIR.to_string(module, generic: true) =~ "func.call"
  end

  test "converts ex.to_word as a pure passthrough", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0(%arg0: i64):
            %0 = "ex.to_word"(%arg0) : (i64) -> !ex.dyn
            %1 = "ex.is_binary"(%0) : (!ex.dyn) -> i64
            "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "func.call"
    assert rendered =~ "ex.term.is_binary"
    refute rendered =~ "arith.shli"
  end

  test "expands multiple integer patterns per clause into an OR condition", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@multi_pattern_module, ctx: ctx)
      |> MLIR.verify!()

    module = ExpandCase.run!(module)
    expanded = MLIR.to_string(module, generic: true)
    refute expanded =~ ~s{"ex.case"}
    assert expanded =~ "arith.ori"

    converted = Plan.run!(Ex.plan(), module)
    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "scf.if"
    assert rendered =~ "arith.cmpi"
  end

  test "expands a clause guard into a narrowed AND condition", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@guard_module, ctx: ctx)
      |> MLIR.verify!()

    module = ExpandCase.run!(module)
    expanded = MLIR.to_string(module, generic: true)
    refute expanded =~ ~s{"ex.case"}
    assert expanded =~ "arith.andi"

    converted = Plan.run!(Ex.plan(), module)
    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "scf.if"
  end

  test "expands a guarded no-pattern clause before more clauses", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.lit"() {value = 1 : i64} : () -> i64
            %2 = "ex.case"(%0) ({
            ^bb0:
              "ex.clause"(%1) {patterns = array<i64>} : (i64) -> ()
              %3 = "ex.lit"() {value = 10 : i64} : () -> i64
              "ex.yield"(%3) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
            ^bb1:
              "ex.clause"() {patterns = array<i64>} : () -> ()
              %4 = "ex.lit"() {value = 20 : i64} : () -> i64
              "ex.yield"(%4) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
            }) {operandSegmentSizes = array<i32: 1>} : (i64) -> i64
            "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    module = ExpandCase.run!(module)
    expanded = MLIR.to_string(module, generic: true)
    # a no-pattern clause matches everything, so the condition is the guard
    assert expanded =~ ~s{predicate = "ne"}

    converted = Plan.run!(Ex.plan(), module)
    assert MLIR.to_string(converted, generic: true) =~ "scf.if"
  end

  test "lowers control flow through to LLVM", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@control_flow_module, ctx: ctx)
      |> MLIR.verify!()
      |> then(&ExpandCase.run!/1)

    converted = Plan.run!(Ex.plan(), module)

    pass_manager = MLIR.CAPI.mlirPassManagerCreate(ctx)

    for pass <- [
          &MLIR.CAPI.mlirCreateConversionArithToLLVMConversionPass/0,
          &MLIR.CAPI.mlirCreateConversionSCFToControlFlowPass/0,
          &MLIR.CAPI.mlirCreateConversionConvertControlFlowToLLVMPass/0,
          &MLIR.CAPI.mlirCreateConversionConvertFuncToLLVMPass/0
        ] do
      MLIR.CAPI.mlirPassManagerAddOwnedPass(pass_manager, pass.())
    end

    assert {:ok, _} = MLIR.PassManager.run(pass_manager, converted)
    rendered = MLIR.to_string(converted, generic: true)
    assert rendered =~ "llvm.br"
    refute rendered =~ "scf.if"
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

  defp term_module!(module_source, ctx) do
    Beaver.Slang.load(ctx, ExDialect)

    MLIR.Module.create!(module_source, ctx: ctx)
    |> MLIR.verify!()
  end

  test "converts term construction and predicates to Zig runtime ABI calls", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.lit"() {value = 2 : i64} : () -> i64
            %2 = "ex.box"(%0) : (i64) -> !ex.dyn
            %3 = "ex.box"(%1) : (i64) -> !ex.dyn
            %4 = "ex.tuple"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %5 = "ex.is_tuple"(%4) : (!ex.dyn) -> i64
            "ex.return"(%5) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.list_cons"
    assert rendered =~ "ex.term.tuple_from_list"
    assert rendered =~ "ex.term.is_tuple"
    # scalar operands are tagged as immediate integer terms before construction
    assert rendered =~ "arith.shli"
  end

  test "converts list, map and binary construction to runtime ABI calls", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.lit"() {value = 2 : i64} : () -> i64
            %2 = "ex.box"(%0) : (i64) -> !ex.dyn
            %3 = "ex.box"(%1) : (i64) -> !ex.dyn
            %4 = "ex.list"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %5 = "ex.map"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %6 = "ex.binary"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %7 = "ex.is_list"(%4) : (!ex.dyn) -> i64
            "ex.return"(%7) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.list_cons"
    assert rendered =~ "ex.term.map_from_list"
    assert rendered =~ "ex.term.binary_from_list"
    assert rendered =~ "ex.term.is_list"
  end

  test "converts term read ops to Zig runtime ABI calls", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.lit"() {value = 2 : i64} : () -> i64
            %2 = "ex.box"(%0) : (i64) -> !ex.dyn
            %3 = "ex.box"(%1) : (i64) -> !ex.dyn
            %4 = "ex.tuple"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %5 = "ex.list"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %6 = "ex.tuple_length"(%4) : (!ex.dyn) -> i64
            %7 = "ex.tuple_get"(%4, %0) : (!ex.dyn, i64) -> !ex.dyn
            %8 = "ex.list_length"(%5) : (!ex.dyn) -> i64
            %9 = "ex.list_head"(%5) : (!ex.dyn) -> !ex.dyn
            %10 = "ex.list_tail"(%5) : (!ex.dyn) -> !ex.dyn
            %11 = "ex.term_eq"(%7, %2) : (!ex.dyn, !ex.dyn) -> i64
            "ex.return"(%11) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.tuple_length"
    assert rendered =~ "ex.term.tuple_get"
    assert rendered =~ "ex.term.list_length"
    assert rendered =~ "ex.term.list_head"
    assert rendered =~ "ex.term.list_tail"
    assert rendered =~ "ex.term.eq"
  end

  test "converts binary read ops to Zig runtime ABI calls", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.lit"() {value = 2 : i64} : () -> i64
            %2 = "ex.box"(%0) : (i64) -> !ex.dyn
            %3 = "ex.box"(%1) : (i64) -> !ex.dyn
            %4 = "ex.binary"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %5 = "ex.binary_length"(%4) : (!ex.dyn) -> i64
            %6 = "ex.binary_get"(%4, %0) : (!ex.dyn, i64) -> !ex.dyn
            %7 = "ex.binary_slice"(%4, %1) : (!ex.dyn, i64) -> !ex.dyn
            "ex.return"(%5) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.binary_length"
    assert rendered =~ "ex.term.binary_get"
    assert rendered =~ "ex.term.binary_slice"
  end

  test "converts utf8 binary read ops to Zig runtime ABI calls", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 0 : i64} : () -> i64
            %1 = "ex.lit"() {value = 195 : i64} : () -> i64
            %2 = "ex.lit"() {value = 169 : i64} : () -> i64
            %3 = "ex.box"(%1) : (i64) -> !ex.dyn
            %4 = "ex.box"(%2) : (i64) -> !ex.dyn
            %5 = "ex.binary"(%3, %4) {operandSegmentSizes = array<i32: 2>} : (!ex.dyn, !ex.dyn) -> !ex.dyn
            %6 = "ex.binary_utf8_width"(%5, %0) : (!ex.dyn, i64) -> i64
            %7 = "ex.binary_utf8_get"(%5, %0) : (!ex.dyn, i64) -> !ex.dyn
            "ex.return"(%6) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.binary_utf8_width"
    assert rendered =~ "ex.term.binary_utf8_get"
  end

  test "converts spawn and scheduler continuation ops to Zig runtime ABI calls", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 0 : i64} : () -> i64
            %1 = "ex.lit"() {value = 1 : i64} : () -> i64
            %2 = "ex.lit"() {value = 2 : i64} : () -> i64
            %3 = "ex.box"(%0) : (i64) -> !ex.dyn
            %4 = "ex.spawn"(%3) : (!ex.dyn) -> !ex.dyn
            %5 = "ex.process_table_reset"() : () -> i64
            %6 = "ex.cont_save"(%1, %2, %0) : (i64, i64, i64) -> i64
            %7 = "ex.receive_cont_save"(%1, %2, %0) : (i64, i64, i64) -> i64
            %8 = "ex.cont_pending"() : () -> i64
            %9 = "ex.cont_active"() : () -> i64
            %10 = "ex.cont_clear"() : () -> i64
            %11 = "ex.cont_load_arg"() : () -> i64
            %12 = "ex.cont_load_acc"() : () -> i64
            %13 = "ex.cont_load_cursor"() : () -> i64
            %14 = "ex.schedule_next"() : () -> i64
            %15 = "ex.mailbox_len"() : () -> i64
            %16 = "ex.mailbox_peek"(%0) : (i64) -> !ex.dyn
            %17 = "ex.mailbox_remove"(%0) : (i64) -> i64
            %18 = "ex.nil_word"() : () -> !ex.dyn
            %19 = "ex.monotonic_time"() : () -> i64
            %20 = "ex.receive_start"() : () -> i64
            %21 = "ex.receive_start_set"(%0) : (i64) -> i64
            %22 = "ex.native_time"() : () -> i64
            %23 = "ex.unique_integer"(%0) : (i64) -> i64
            %24 = "ex.current_entry"() : () -> i64
            %25 = "ex.process_done"(%0) : (i64) -> i64
            %26 = "ex.processes_runnable"() : () -> i64
            %27 = "ex.process_result"(%4) : (!ex.dyn) -> i64
            %28 = "ex.func_addr"() {sym_name = "actor_step"} : () -> ((i64) -> i64)
            %29 = "ex.worker_run"(%1, %28) : (i64, (i64) -> i64) -> i64
            %30 = "ex.process_wait"(%0) : (i64) -> i64
            "ex.return"(%29) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.spawn"
    assert rendered =~ "ex.term.process_table_reset"
    assert rendered =~ "ex.term.cont_save"
    assert rendered =~ "ex.term.receive_cont_save"
    assert rendered =~ "ex.term.cont_pending"
    assert rendered =~ "ex.term.cont_active"
    assert rendered =~ "ex.term.cont_clear"
    assert rendered =~ "ex.term.cont_load_arg"
    assert rendered =~ "ex.term.cont_load_acc"
    assert rendered =~ "ex.term.cont_load_cursor"
    assert rendered =~ "ex.term.schedule_next"
    assert rendered =~ "ex.term.mailbox_len"
    assert rendered =~ "ex.term.mailbox_peek"
    assert rendered =~ "ex.term.mailbox_remove"
    assert rendered =~ "ex.term.nil"
    assert rendered =~ "ex.term.monotonic_time"
    assert rendered =~ "ex.term.receive_start"
    assert rendered =~ "ex.term.receive_start_set"
    assert rendered =~ "ex.term.native_time"
    assert rendered =~ "ex.term.unique_integer"
    assert rendered =~ "ex.term.current_entry"
    assert rendered =~ "ex.term.process_done"
    assert rendered =~ "ex.term.processes_runnable"
    assert rendered =~ "ex.term.process_result"
    assert rendered =~ "ex.term.worker_run"
    assert rendered =~ "ex.term.process_wait"
  end

  test "passes nested term operands through without re-tagging", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.dyn
            %2 = "ex.tuple"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.dyn) -> !ex.dyn
            %3 = "ex.tuple"(%2) {operandSegmentSizes = array<i32: 1>} : (!ex.dyn) -> !ex.dyn
            %4 = "ex.is_tuple"(%3) : (!ex.dyn) -> i64
            "ex.return"(%4) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.tuple_from_list"
    assert rendered =~ "ex.term.is_tuple"
  end

  test "rejects ex.map with an odd number of entries", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.dyn
            %2 = "ex.map"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.dyn) -> !ex.dyn
            "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:error, %MLIR.Conversion.Error{} = error} = Plan.run(Ex.plan(), module)
    assert Exception.message(error) =~ "even number"
  end
end
