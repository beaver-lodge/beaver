defmodule ExConversionTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Conversion.Ex
  alias Beaver.MLIR.Conversion.Ex.Stage0
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex, as: ExDialect
  alias Beaver.MLIR.Dialect.Ex.ExpandCase
  alias Beaver.MLIR.Dialect.Ex.MaterializeBoundVariables

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
      %3 = "ex.call"(%0, %1) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i64, i64) -> !ex.term
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

  @term_case_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = -1 : i64} : () -> i64
      %1 = "ex.box"(%0) : (i64) -> !ex.term
      %2 = "ex.case"(%1) ({
      ^bb0:
        "ex.clause"() {patterns = array<i64: 0>} : () -> ()
        %3 = "ex.lit"() {value = 10 : i64} : () -> i64
        "ex.yield"(%3) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      ^bb1:
        "ex.clause"() {patterns = array<i64: 1>} : () -> ()
        %4 = "ex.lit"() {value = 20 : i64} : () -> i64
        "ex.yield"(%4) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      ^bb2:
        "ex.clause"() {patterns = array<i64: -1>} : () -> ()
        %5 = "ex.lit"() {value = 30 : i64} : () -> i64
        "ex.yield"(%5) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      ^bb3:
        "ex.clause"() {patterns = array<i64>} : () -> ()
        %6 = "ex.lit"() {value = 40 : i64} : () -> i64
        "ex.yield"(%6) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
      }) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> i64
      "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
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

  test "uses native type maps and hot scalar patterns", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@bind_module, ctx: ctx)
      |> MLIR.verify!()
      |> MaterializeBoundVariables.run!()

    assert {_converted, receipt} = Plan.profile!(Ex.plan(), module)

    refute Enum.any?(receipt["callbacks"], &(&1["kind"] == "convert_type"))

    pattern_callbacks =
      Enum.filter(receipt["callbacks"], &(&1["kind"] == "conversion_pattern"))

    assert Enum.map(pattern_callbacks, &{&1["root"], &1["count"]}) == [
             {"ex.func", 1},
             {"ex.return", 1}
           ]

    declaration = Plan.declaration(Ex.plan())
    assert %{kind: :add_pattern_population, version: "1.0"} in declaration.entries

    moved_roots =
      ~w(ex.lit ex.box ex.to_word ex.unbox ex.yield ex.add ex.sub ex.mul ex.div ex.rem ex.cmp ex.if)

    refute Enum.any?(declaration.entries, fn entry ->
             entry[:kind] == :add_conversion_pattern and entry[:root] in moved_roots
           end)
  end

  test "freezes the machine-readable C++ Stage 0 boundary" do
    manifest = Stage0.manifest()

    assert manifest["schema_version"] == 1
    assert manifest["provider"] == "cpp-bootstrap"
    assert manifest["entrypoint"] == "beaverPopulateExScalarConversionPatterns"
    assert manifest["identity_digest"] == Stage0.identity_digest()

    assert Stage0.identity_digest() ==
             "sha256:6e5d22d6e59047a2875c55104427a343affd23fdfd867a5d988fb34f15e64d4c"

    assert Stage0.roots() ==
             ~w(ex.add ex.box ex.cmp ex.div ex.if ex.lit ex.mul ex.rem ex.sub ex.to_word ex.unbox ex.yield)

    assert Enum.map(manifest["patterns"], & &1["root"]) == Stage0.roots()

    declaration = Plan.declaration(Ex.plan())

    for root <- ~w(ex.binary ex.binary_part ex.term_eq) do
      assert Enum.any?(declaration.entries, fn entry ->
               entry[:kind] == :add_conversion_pattern and entry[:root] == root
             end)
    end
  end

  test "exposes a stable Ex dialect schema identity" do
    assert ExDialect.schema_digest() ==
             "sha256:86f6e42a47a8a062fba31ccc75afaeb8c7b7e749c0df57419a518fd2a3fc5e67"
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
            %1 = "ex.call"(%0, %0) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i64, i64) -> !ex.term
            "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> ()
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
    refute rendered =~ "!ex.term"
  end

  test "allows heterogeneous-typed ex.call arguments", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.term
            %2 = "ex.call"(%1, %0) {callee = "f", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (!ex.term, i64) -> i64
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
            %0 = "ex.to_word"(%arg0) : (i64) -> !ex.term
            %1 = "ex.is_binary"(%0) : (!ex.term) -> i64
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

  test "expands integer patterns against a term scrutinee with strict term equality", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(@term_case_module, ctx: ctx)
      |> MLIR.verify!()
      |> ExpandCase.run!()
      |> MLIR.verify!()

    expanded = MLIR.to_string(module, generic: true)
    assert length(Regex.scan(~r/"ex\.term_eq"/, expanded)) == 3
    refute expanded =~ ~r/"ex\.cmp"\([^\n]*\) : \(!ex\.term, i64\)/

    converted = Plan.run!(Ex.plan(), module)
    assert MLIR.verify?(converted)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ ~s{"ex.term_eq"}
    assert rendered =~ "ex.term.eq"
  end

  test "rejects integer patterns against an unsupported scrutinee type", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "arith.constant"() {value = 1.0 : f64} : () -> f64
            %1 = "ex.case"(%0) ({
            ^bb0:
              "ex.clause"() {patterns = array<i64: 1>} : () -> ()
              %2 = "ex.lit"() {value = 10 : i64} : () -> i64
              "ex.yield"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
            }) {operandSegmentSizes = array<i32: 1>} : (f64) -> i64
            "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx: ctx
      )
      |> MLIR.verify!()

    assert_raise ArgumentError,
                 "ex.case integer patterns require an i64 or !ex.term scrutinee, got: f64",
                 fn -> ExpandCase.run!(module) end
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
            %2 = "ex.box"(%0) : (i64) -> !ex.term
            %3 = "ex.box"(%1) : (i64) -> !ex.term
            %4 = "ex.tuple"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %5 = "ex.is_tuple"(%4) : (!ex.term) -> i64
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

  test "converts typed raises to a distinct runtime ABI call", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 7 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.term
            %2 = "ex.raise"(%1, %0) : (!ex.term, i64) -> !ex.term
            "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.raise"
    refute rendered =~ "ex.term.throw"
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
            %2 = "ex.box"(%0) : (i64) -> !ex.term
            %3 = "ex.box"(%1) : (i64) -> !ex.term
            %4 = "ex.list"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %19 = "ex.list_flatten"(%4) : (!ex.term) -> !ex.term
            %24 = "ex.list_insert_at"(%4, %0, %3) : (!ex.term, i64, !ex.term) -> !ex.term
            %5 = "ex.map"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %6 = "ex.map_put"(%5, %2, %3) : (!ex.term, !ex.term, !ex.term) -> !ex.term
            %13 = "ex.map_fetch"(%6, %2) : (!ex.term, !ex.term) -> !ex.term
            %14 = "ex.enumerable_into_map"(%4, %6) : (!ex.term, !ex.term) -> !ex.term
            %15 = "ex.enumerable_intersperse"(%4, %3) : (!ex.term, !ex.term) -> !ex.term
            %16 = "ex.enumerable_map_term_fun"(%4, %0) : (!ex.term, i64) -> !ex.term
            %18 = "ex.enumerable_map_term_fun_c"(%4, %0, %0, %0, %0, %0) : (!ex.term, i64, i64, i64, i64, i64) -> !ex.term
            %17 = "ex.enumerable_flat_map_term_fun"(%4, %0) : (!ex.term, i64) -> !ex.term
            %7 = "ex.binary"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %8 = "ex.binary_from_list"(%4) : (!ex.term) -> !ex.term
            %12 = "ex.iodata_to_binary"(%4) : (!ex.term) -> !ex.term
            %9 = "ex.float_lit"(%0) : (i64) -> !ex.term
            %23 = "ex.bigint_lit"(%7) : (!ex.term) -> !ex.term
            %10 = "ex.string_to_float"(%7) : (!ex.term) -> !ex.term
            %21 = "ex.string_to_atom"(%7) : (!ex.term) -> !ex.term
            %22 = "ex.string_to_existing_atom"(%7) : (!ex.term) -> !ex.term
            %20 = "ex.float_to_binary_short"(%9) : (!ex.term) -> !ex.term
            %11 = "ex.is_list"(%4) : (!ex.term) -> i64
            "ex.return"(%11) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.list_cons"
    assert rendered =~ "ex.term.list_flatten"
    assert rendered =~ "ex.term.list_insert_at"
    assert rendered =~ "ex.term.map_from_list"
    assert rendered =~ "ex.term.map_put"
    assert rendered =~ "ex.term.map_fetch"
    assert rendered =~ "ex.term.enumerable_into_map"
    assert rendered =~ "ex.term.enumerable_intersperse"
    assert rendered =~ "ex.term.enumerable_map_term_fun"
    assert rendered =~ "ex.term.enumerable_map_term_fun_c"
    assert rendered =~ "ex.term.enumerable_flat_map_term_fun"
    assert rendered =~ "ex.term.binary_from_list"
    assert rendered =~ "ex.term.iodata_to_binary"
    assert rendered =~ "ex.term.float_lit"
    assert rendered =~ "ex.term.bigint_lit"
    assert rendered =~ "ex.term.string_to_float"
    assert rendered =~ "ex.term.string_to_atom"
    assert rendered =~ "ex.term.string_to_existing_atom"
    assert rendered =~ "ex.term.float_to_binary_short"
    assert rendered =~ "ex.term.is_list"
  end

  test "reuses the list_cons declaration while lowering a wide binary", %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)

    module =
      mlir ctx: ctx do
        module do
          ExDialect.func sym_name: ~a/wide_binary/s do
            region do
              block do
                segments =
                  Enum.map(0..31, fn index ->
                    value =
                      ExDialect.lit(value: Attribute.integer(Type.i64(), index)) >>> Type.i64()

                    ExDialect.box(value: value) >>> ExDialect.term()
                  end)

                binary =
                  ExDialect.binary(segments: segments, operandSegmentSizes: :infer) >>>
                    ExDialect.term()

                ExDialect.return(binary) >>> []
              end
            end
          end >>> []
        end
      end
      |> MLIR.verify!()

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)

    assert length(Regex.scan(~r/sym_name = "ex\.term\.list_cons"/, rendered)) == 1
    assert length(Regex.scan(~r/sym_name = "ex\.term\.binary_from_list"/, rendered)) == 1
    assert length(Regex.scan(~r/callee = @ex\.term\.list_cons/, rendered)) == 32
    assert length(Regex.scan(~r/callee = @ex\.term\.binary_from_list/, rendered)) == 1
    assert MLIR.verify!(module) == module
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
            %2 = "ex.box"(%0) : (i64) -> !ex.term
            %3 = "ex.box"(%1) : (i64) -> !ex.term
            %4 = "ex.tuple"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %5 = "ex.list"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %6 = "ex.tuple_length"(%4) : (!ex.term) -> i64
            %7 = "ex.tuple_get"(%4, %0) : (!ex.term, i64) -> !ex.term
            %8 = "ex.list_length"(%5) : (!ex.term) -> i64
            %9 = "ex.list_head"(%5) : (!ex.term) -> !ex.term
            %10 = "ex.list_tail"(%5) : (!ex.term) -> !ex.term
            %11 = "ex.term_eq"(%7, %2) : (!ex.term, !ex.term) -> i64
            %12 = "ex.term_eq_loose"(%7, %2) : (!ex.term, !ex.term) -> i64
            %17 = "ex.integer_compare"(%2, %3) : (!ex.term, !ex.term) -> i64
            %13 = "ex.string_printable"(%2) : (!ex.term) -> i64
            %14 = "ex.binary_quote"(%2) : (!ex.term) -> !ex.term
            %15 = "ex.int_to_hex"(%2) : (!ex.term) -> !ex.term
            %16 = "ex.int_to_string_base"(%2, %1) : (!ex.term, i64) -> !ex.term
            "ex.return"(%13) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.tuple_length"
    assert rendered =~ "ex.term.tuple_get"
    assert rendered =~ "ex.term.integer_compare"
    assert rendered =~ "ex.term.list_length"
    assert rendered =~ "ex.term.list_head"
    assert rendered =~ "ex.term.list_tail"
    assert rendered =~ "ex.term.eq"
    assert rendered =~ "ex.term.eq_loose"

    assert length(Regex.scan(~r/sym_name = "ex\.term\.eq"/, rendered)) == 1

    assert rendered =~
             ~s|function_type = (i64, i64) -> i64, sym_name = "ex.term.eq_loose"|

    assert rendered =~ "ex.term.string_printable"
    assert rendered =~ "ex.term.binary_quote"
    assert rendered =~ "ex.term.int_to_hex"

    assert rendered =~
             ~s|function_type = (i64, i64) -> i64, sym_name = "ex.term.int_to_string_base"|
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
            %2 = "ex.box"(%0) : (i64) -> !ex.term
            %3 = "ex.box"(%1) : (i64) -> !ex.term
            %4 = "ex.binary"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %5 = "ex.binary_length"(%4) : (!ex.term) -> i64
            %6 = "ex.binary_get"(%4, %0) : (!ex.term, i64) -> !ex.term
            %7 = "ex.binary_slice"(%4, %1) : (!ex.term, i64) -> !ex.term
            %8 = "ex.binary_part"(%4, %2, %3) : (!ex.term, !ex.term, !ex.term) -> !ex.term
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
    assert rendered =~ "ex.term.binary_part"

    assert rendered =~
             ~s|function_type = (i64, i64, i64) -> i64, sym_name = "ex.term.binary_part"|
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
            %3 = "ex.box"(%1) : (i64) -> !ex.term
            %4 = "ex.box"(%2) : (i64) -> !ex.term
            %5 = "ex.binary"(%3, %4) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %6 = "ex.binary_utf8_width"(%5, %0) : (!ex.term, i64) -> i64
            %7 = "ex.binary_utf8_get"(%5, %0) : (!ex.term, i64) -> !ex.term
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
            %3 = "ex.box"(%0) : (i64) -> !ex.term
            %4 = "ex.spawn"(%3) : (!ex.term) -> !ex.term
            %5 = "ex.runtime_create"() : () -> i64
            %6 = "ex.runtime_enter"(%5) : (i64) -> i64
            %runtime_leave = "ex.runtime_leave"() : () -> i64
            %runtime_destroy = "ex.runtime_destroy"(%5) : (i64) -> i64
            %result_handle = "ex.result_create"(%5, %0) : (i64, i64) -> i64
            %result_kind = "ex.result_root_kind"(%result_handle) : (i64) -> i64
            %result_word = "ex.result_root_word"(%result_handle) : (i64) -> i64
            %term_kind = "ex.result_term_kind"(%result_handle, %result_word) : (i64, i64) -> i64
            %atom_name = "ex.result_atom_name"(%result_handle, %result_word) : (i64, i64) -> i64
            %term_len = "ex.result_term_length"(%result_handle, %result_word) : (i64, i64) -> i64
            %term_item = "ex.result_term_get"(%result_handle, %result_word, %0) : (i64, i64, i64) -> i64
            %exported = "ex.term_export"(%result_handle, %result_word) : (i64, i64) -> i64
            %exported_clone = "ex.exported_clone"(%exported) : (i64) -> i64
            %exported_len = "ex.exported_length"(%exported) : (i64) -> i64
            %exported_byte = "ex.exported_get"(%exported, %0) : (i64, i64) -> i64
            %term_handle = "ex.term_import"(%5, %exported) : (i64, i64) -> i64
            %exported_again = "ex.term_handle_export"(%term_handle) : (i64) -> i64
            %term_handle_destroy = "ex.term_handle_destroy"(%term_handle) : (i64) -> i64
            %exported_destroy = "ex.exported_destroy"(%exported_clone) : (i64) -> i64
            %result_destroy = "ex.result_destroy"(%result_handle) : (i64) -> i64
            %7 = "ex.process_table_reset"(%1) : (i64) -> i64
            %8 = "ex.cont_save"(%1, %2, %0) : (i64, i64, i64) -> i64
            %9 = "ex.receive_cont_save"(%1, %2, %0) : (i64, i64, i64) -> i64
            %10 = "ex.cont_pending"() : () -> i64
            %11 = "ex.cont_active"() : () -> i64
            %12 = "ex.cont_clear"() : () -> i64
            %13 = "ex.cont_load_arg"() : () -> i64
            %14 = "ex.cont_load_acc"() : () -> i64
            %15 = "ex.cont_load_cursor"() : () -> i64
            %16 = "ex.schedule_next"() : () -> i64
            %17 = "ex.mailbox_len"() : () -> i64
            %18 = "ex.mailbox_peek"(%0) : (i64) -> !ex.term
            %19 = "ex.mailbox_remove"(%0) : (i64) -> i64
            %20 = "ex.nil_word"() : () -> !ex.term
            %21 = "ex.monotonic_time"() : () -> i64
            %22 = "ex.receive_start"() : () -> i64
            %23 = "ex.receive_start_set"(%0) : (i64) -> i64
            %24 = "ex.native_time"() : () -> i64
            %25 = "ex.unique_integer"(%0) : (i64) -> i64
            %26 = "ex.current_entry"() : () -> i64
            %27 = "ex.process_done"(%0) : (i64) -> i64
            %28 = "ex.processes_runnable"() : () -> i64
            %29 = "ex.process_result"(%4) : (!ex.term) -> i64
            %30 = "ex.func_addr"() {sym_name = "actor_step"} : () -> ((i64) -> i64)
            %31 = "ex.worker_run"(%1, %30) : (i64, (i64) -> i64) -> i64
            %32 = "ex.process_wait"(%0) : (i64) -> i64
            "ex.return"(%31) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.spawn"
    assert rendered =~ "ex.term.runtime_create"
    assert rendered =~ "ex.term.runtime_enter"
    assert rendered =~ "ex.term.runtime_leave"
    assert rendered =~ "ex.term.runtime_destroy"
    assert rendered =~ "ex.term.result_create"
    assert rendered =~ "ex.term.result_destroy"
    assert rendered =~ "ex.term.result_root_kind"
    assert rendered =~ "ex.term.result_root_word"
    assert rendered =~ "ex.term.result_term_kind"
    assert rendered =~ "ex.term.result_atom_name"
    assert rendered =~ "ex.term.result_term_length"
    assert rendered =~ "ex.term.result_term_get"
    assert rendered =~ "ex.term.export"
    assert rendered =~ "ex.term.import"
    assert rendered =~ "ex.term.exported_clone"
    assert rendered =~ "ex.term.exported_destroy"
    assert rendered =~ "ex.term.exported_length"
    assert rendered =~ "ex.term.exported_get"
    assert rendered =~ "ex.term.handle_export"
    assert rendered =~ "ex.term.handle_destroy"

    assert rendered =~
             ~s|"func.func"() <{function_type = (i64, i64, i64) -> i64, sym_name = "ex.term.result_term_get"|

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

  test "converts process supervision ops to Zig runtime ABI calls", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 0 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.term
            %2 = "ex.process_exit"(%1) : (!ex.term) -> !ex.term
            %3 = "ex.process_exit_reason"(%1) : (!ex.term) -> !ex.term
            %4 = "ex.process_trap_exit"(%0) : (i64) -> i64
            %5 = "ex.process_dictionary_get"(%1, %1) : (!ex.term, !ex.term) -> !ex.term
            %6 = "ex.process_dictionary_put"(%1, %5) : (!ex.term, !ex.term) -> !ex.term
            %7 = "ex.link"(%1, %1, %1) : (!ex.term, !ex.term, !ex.term) -> !ex.term
            %8 = "ex.unlink"(%1) : (!ex.term) -> i64
            %9 = "ex.exit"(%1, %1, %1, %1) : (!ex.term, !ex.term, !ex.term, !ex.term) -> !ex.term
            %10 = "ex.monitor"(%1, %1, %1, %1) : (!ex.term, !ex.term, !ex.term, !ex.term) -> !ex.term
            %11 = "ex.demonitor"(%10) : (!ex.term) -> i64
            "ex.return"(%11) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)

    for symbol <- ~w(
      process_exit process_exit_reason process_trap_exit process_dictionary_get
      process_dictionary_put link unlink exit monitor demonitor
    ) do
      assert rendered =~ "ex.term.#{symbol}"
    end
  end

  test "passes nested term operands through without re-tagging", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.term
            %2 = "ex.tuple"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
            %3 = "ex.tuple"(%2) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
            %4 = "ex.is_tuple"(%3) : (!ex.term) -> i64
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

  test "keeps legacy closure lowering and adds arity-carrying closure reads", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 7 : i64} : () -> i64
            %1 = "ex.make_fun"(%0) {fn_idx = 1 : i64, env_len = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i64) -> !ex.term
            %2 = "ex.make_fun_with_arity"(%0) {fn_idx = 2 : i64, arity = 1 : i64, env_len = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i64) -> !ex.term
            %3 = "ex.fun_arity"(%2) : (!ex.term) -> i64
            %4 = "ex.make_fun_with_signature"(%0) {fn_idx = 3 : i64, arity = 1 : i64, result_mode = 1 : i64, env_len = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i64) -> !ex.term
            %5 = "ex.fun_result_mode"(%4) : (!ex.term) -> i64
            "ex.return"(%5) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.make_fun"
    assert rendered =~ "ex.term.make_fun_with_arity"
    assert rendered =~ "ex.term.make_fun_with_signature"
    assert rendered =~ "ex.term.fun_arity"
    assert rendered =~ "ex.term.fun_result_mode"
    assert rendered =~ "(i64, i64, i64, i64, i64, i64) -> i64"
    assert rendered =~ "(i64, i64, i64, i64, i64, i64, i64) -> i64"
    assert rendered =~ "(i64, i64, i64, i64, i64, i64, i64, i64) -> i64"
  end

  test "rejects ex.map with an odd number of entries", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 1 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.term
            %2 = "ex.map"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
            "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:error, %MLIR.Conversion.Error{} = error} = Plan.run(Ex.plan(), module)
    assert Exception.message(error) =~ "even number"
  end

  test "lowers abstract !ex.term types to scalar words", %{ctx: ctx} do
    module =
      term_module!(
        ~S"""
        module {
          "ex.func"() ({
          ^bb0:
            %0 = "ex.lit"() {value = 42 : i64} : () -> i64
            %1 = "ex.box"(%0) : (i64) -> !ex.term
            %2 = "ex.tuple"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
            %3 = "ex.list"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
            %4 = "ex.map"(%1, %2) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
            %5 = "ex.binary"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
            %6 = "ex.send"(%1, %2) : (!ex.term, !ex.term) -> !ex.term
            %7 = "ex.to_int"(%1) : (!ex.term) -> i64
            "ex.return"(%7) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
          }) {sym_name = "main"} : () -> ()
        }
        """,
        ctx
      )

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "ex.term.tuple_from_list"
    assert rendered =~ "ex.term.map_from_list"
    assert rendered =~ "ex.term.binary_from_list"
    assert rendered =~ "ex.term.send"
  end
end
