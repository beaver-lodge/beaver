defmodule ExConversionTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
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
      %3 = "ex.call"(%0, %1) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i64, i64) -> i64
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

  setup %{ctx: ctx} do
    Beaver.Slang.load(ctx, ExDialect)
    :ok
  end

  test "lowers the ex scalar subset to func/arith and llvm", %{ctx: ctx} do
    module =
      parse!(ctx, @scalar_module)

    plan = Ex.plan()
    assert {:ok, ^module, _diagnostics} = Plan.run(plan, module)

    rendered = MLIR.to_string(module, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "arith.addi"
    assert rendered =~ "func.call"
    assert rendered =~ "func.return"

    module
    |> Beaver.Composer.append("convert-arith-to-llvm,convert-func-to-llvm")
    |> Beaver.Composer.run!()

    rendered_llvm = MLIR.to_string(module)
    assert rendered_llvm =~ "llvm.func"
    assert rendered_llvm =~ "llvm.add"
  end

  test "materializes ex.var/ex.bind into SSA", %{ctx: ctx} do
    module =
      parse!(ctx, @bind_module)
      |> MaterializeBoundVariables.run!()

    assert converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex.var"
    refute rendered =~ "ex.bind"
    assert rendered =~ "arith.constant"
    assert rendered =~ "func.return"
  end

  test "uses native type maps and hot scalar patterns", %{ctx: ctx} do
    module =
      parse!(ctx, @bind_module)
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
      ~w(ex.lit ex.yield ex.add ex.sub ex.mul ex.div ex.rem ex.cmp ex.if)

    refute Enum.any?(declaration.entries, fn entry ->
             entry[:kind] == :add_conversion_pattern and entry[:root] in moved_roots
           end)
  end

  test "freezes the machine-readable C++ Stage 0 boundary" do
    manifest = Stage0.manifest()

    assert manifest["schema_version"] == 2
    assert manifest["provider"] == "cpp-bootstrap"
    assert manifest["entrypoint"] == "beaverPopulateExScalarConversionPatterns"
    assert manifest["identity_digest"] == Stage0.identity_digest()

    assert Stage0.identity_digest() ==
             "sha256:e483a265a7a42dec9644bfe1c06b06f9605607b2ddee417ac331421230ebafac"

    assert Stage0.roots() ==
             ~w(ex.add ex.cmp ex.div ex.if ex.lit ex.mul ex.rem ex.sub ex.yield)

    assert Enum.map(manifest["patterns"], & &1["root"]) == Stage0.roots()

    declaration = Plan.declaration(Ex.plan())

    for root <- ~w(ex.binary ex.binary_part ex.term_eq) do
      refute Enum.any?(declaration.entries, fn entry ->
               entry[:kind] == :add_conversion_pattern and entry[:root] == root
             end)
    end
  end

  test "exposes a stable Ex dialect schema identity" do
    assert ExDialect.schema_digest() ==
             "sha256:dd1ecfddf8525a4270066c6a60a055da1956c39042b2c16414d29f9c17af44d2"
  end

  test "fails explicitly on a bare ex.var", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
      module {
        "ex.func"() ({
        ^bb0:
          %0 = "ex.var"() {name = "x"} : () -> !ex.unbound
          "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
        }) {sym_name = "main"} : () -> ()
      }
      """)

    module = MaterializeBoundVariables.run!(module)

    assert {:error, %MLIR.Conversion.Error{}} = Plan.run(Ex.plan(), module)
  end

  test "rejects implicit conversion of abstract terms to tagged words", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
      module {
        "ex.func"() ({
        ^bb0:
          %0 = "ex.lit"() {value = 1 : i64} : () -> i64
          %1 = "ex.call"(%0, %0) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i64, i64) -> !ex.term
          "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> ()
        }) {sym_name = "main"} : () -> ()
      }
      """)

    assert {:error, %MLIR.Conversion.Error{}} = Plan.run(Ex.plan(), module)
  end

  test "lowers sub/mul and function arguments", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
      module {
        "ex.func"() ({
        ^bb0(%arg0: i64, %arg1: i64):
          %0 = "ex.mul"(%arg0, %arg1) : (i64, i64) -> i64
          %1 = "ex.sub"(%0, %arg0) : (i64, i64) -> i64
          "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
        }) {sym_name = "main"} : () -> ()
      }
      """)

    converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ ~s{function_type = (i64, i64) -> i64, sym_name = "main"}
    assert rendered =~ "^bb0(%arg0: i64, %arg1: i64)"
    assert rendered =~ "arith.muli"
    assert rendered =~ "arith.subi"
  end

  test "lowers ex control flow ops to scf", %{ctx: ctx} do
    module =
      parse!(ctx, @control_flow_module)

    converted = Plan.run!(Ex.plan(), module)

    rendered = MLIR.to_string(converted, generic: true)
    refute rendered =~ "ex."
    assert rendered =~ "arith.cmpi"
    assert rendered =~ "scf.if"
    assert rendered =~ "scf.yield"
  end

  test "rejects unknown ex.cmp predicates", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
      module {
        "ex.func"() ({
        ^bb0:
          %0 = "ex.lit"() {value = 1 : i64} : () -> i64
          %1 = "ex.lit"() {value = 2 : i64} : () -> i64
          %2 = "ex.cmp"(%0, %1) {predicate = "bogus"} : (i64, i64) -> i64
          "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
        }) {sym_name = "main"} : () -> ()
      }
      """)

    assert {:error, %MLIR.Conversion.Error{} = error} = Plan.run(Ex.plan(), module)
    assert Exception.message(error) =~ "ex.cmp"
  end

  test "expands ex.case into nested ex.if and lowers to scf", %{ctx: ctx} do
    module =
      parse!(ctx, @case_module)

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
      parse!(ctx, ~S"""
      module {
        "ex.func"() ({
        ^bb0:
          %0 = "ex.lit"() {value = 1 : i64} : () -> i64
          %1 = "ex.call"(%0) {callee = "id", arity = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>} : (i64) -> i64
          "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
        }) {sym_name = "main"} : () -> ()
      }
      """)

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ "func.call"
    refute rendered =~ "!ex.term"
  end

  test "allows heterogeneous-typed ex.call arguments", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
      module {
        "ex.func"() ({
        ^bb0:
          %0 = "ex.lit"() {value = 1 : i64} : () -> i64
          %1 = arith.constant 2 : i32
          %2 = "ex.call"(%1, %0) {callee = "f", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i32, i64) -> i64
          "ex.return"(%2) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
        }) {sym_name = "main"} : () -> ()
      }
      """)

    assert {:ok, _module, []} = Plan.run(Ex.plan(), module)
    assert MLIR.to_string(module, generic: true) =~ "func.call"
  end

  test "expands multiple integer patterns per clause into an OR condition", %{ctx: ctx} do
    module =
      parse!(ctx, @multi_pattern_module)

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
    module =
      parse!(ctx, @term_case_module)
      |> ExpandCase.run!()
      |> MLIR.verify!()

    expanded = MLIR.to_string(module, generic: true)
    assert length(Regex.scan(~r/"ex\.term_eq"/, expanded)) == 3
    refute expanded =~ ~r/"ex\.cmp"\([^\n]*\) : \(!ex\.term, i64\)/

    assert {:error, %MLIR.Conversion.Error{}} = Plan.run(Ex.plan(), module)
  end

  test "rejects integer patterns against an unsupported scrutinee type", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
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
      """)

    assert_raise ArgumentError,
                 "ex.case integer patterns require an i64 or !ex.term scrutinee, got: f64",
                 fn -> ExpandCase.run!(module) end
  end

  test "expands a clause guard into a narrowed AND condition", %{ctx: ctx} do
    module =
      parse!(ctx, @guard_module)

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
    module =
      parse!(ctx, ~S"""
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
      """)

    module = ExpandCase.run!(module)
    expanded = MLIR.to_string(module, generic: true)
    # a no-pattern clause matches everything, so the condition is the guard
    assert expanded =~ ~s{predicate = "ne"}

    converted = Plan.run!(Ex.plan(), module)
    assert MLIR.to_string(converted, generic: true) =~ "scf.if"
  end

  test "lowers control flow through to LLVM", %{ctx: ctx} do
    module =
      parse!(ctx, @control_flow_module)
      |> ExpandCase.run!()

    converted = Plan.run!(Ex.plan(), module)

    converted
    |> Beaver.Composer.append(
      "convert-arith-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,convert-func-to-llvm"
    )
    |> Beaver.Composer.run!()

    rendered = MLIR.to_string(converted, generic: true)
    assert rendered =~ "llvm.br"
    refute rendered =~ "scf.if"
  end

  test "rejects a case without a catch-all clause", %{ctx: ctx} do
    module =
      parse!(ctx, ~S"""
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
      """)

    assert_raise ArgumentError, ~r/without a catch-all/, fn ->
      ExpandCase.run!(module)
    end
  end

  defp parse!(ctx, source), do: MLIR.Module.create!(source, ctx: ctx) |> MLIR.verify!()
end
