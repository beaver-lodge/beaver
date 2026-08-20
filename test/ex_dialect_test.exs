defmodule ExDialectTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.Ex
  use Beaver

  @moduletag :smoke

  @scalar_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 42 : i64} : () -> i64
      %1 = "ex.var"() {name = "x"} : () -> !ex.unbound
      %2 = "ex.bind"(%1, %0) : (!ex.unbound, i64) -> !ex.bound
      %3 = "ex.add"(%0, %0) : (i64, i64) -> i64
      %4 = "ex.call"(%0, %0) {callee = "add", arity = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>} : (i64, i64) -> !ex.term
      "ex.return"(%3) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
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
      }) {operandSegmentSizes = array<i32: 1>} : (i64) -> i64
      "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
    }) {sym_name = "main"} : () -> ()
  }
  """

  @term_module ~S"""
  module {
    "ex.func"() ({
    ^bb0:
      %0 = "ex.lit"() {value = 1 : i64} : () -> i64
      %1 = "ex.lit"() {value = 2 : i64} : () -> i64
      %2 = "ex.box"(%0) : (i64) -> !ex.term
      %3 = "ex.box"(%1) : (i64) -> !ex.term
      %4 = "ex.tuple"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
      %5 = "ex.list"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
      %6 = "ex.map"(%2, %3) {operandSegmentSizes = array<i32: 2>} : (!ex.term, !ex.term) -> !ex.term
      %7 = "ex.binary"(%2) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> !ex.term
      %8 = "ex.is_tuple"(%4) : (!ex.term) -> i64
      %9 = "ex.is_list"(%5) : (!ex.term) -> i64
      %10 = "ex.is_map"(%6) : (!ex.term) -> i64
      %11 = "ex.is_binary"(%7) : (!ex.term) -> i64
      %12 = "ex.is_integer"(%0) : (i64) -> i64
      %13 = "ex.is_atom"(%0) : (i64) -> i64
      "ex.return"(%8) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
    }) {sym_name = "main"} : () -> ()
  }
  """

  test "emits a named IRDL schema for the scalar subset", %{ctx: ctx} do
    schema = Ex.__slang_dialect__(ctx) |> MLIR.verify!()
    rendered = MLIR.to_string(schema, generic: true)

    assert rendered =~ ~s{"irdl.all_of"}
    assert rendered =~ ~s{base_name = "!builtin.integer"}
    assert rendered =~ ~s{base_name = "#builtin.integer"}
    assert rendered =~ ~s{base_name = "#builtin.string"}
    assert rendered =~ ~s{base_name = "#builtin.string"}
    assert rendered =~ ~s{base_ref = @ex::@unbound}
    assert rendered =~ ~s{base_ref = @ex::@bound}
    assert rendered =~ ~s{base_ref = @ex::@term}
    assert rendered =~ ~s{"irdl.attributes"}
    assert rendered =~ ~s{attributeValueNames = ["value"]}
    assert rendered =~ ~s{attributeValueNames = ["name"]}
    assert rendered =~ ~s{attributeValueNames = ["callee", "arity"]}
    assert rendered =~ ~s{attributeValueNames = ["predicate"]}
    assert rendered =~ ~s{attributeValueNames = ["patterns"]}
    assert rendered =~ ~s{attributeValueNames = ["sym_name"]}
    assert rendered =~ ~s{names = ["result"]}
    assert rendered =~ ~s{"irdl.regions"}
    assert rendered =~ ~s{names = ["body"]}
    assert rendered =~ ~s{variadicity_array[variadic]}
    assert rendered =~ ~s{variadicity_array[optional]}
  end

  test "loads the dialect and attaches dynamic traits", %{ctx: ctx} do
    assert Ex.__slang_traits__() == [
             {"yield", [:terminator]},
             {"func", [:isolated_from_above]},
             {"return", [:terminator]}
           ]

    assert Ex |> then(&Beaver.Slang.load(ctx, &1)) |> MLIR.LogicalResult.success?()
    assert MLIR.Context.terminator?(ctx, "ex.return")
  end

  test "constructs the scalar subset with text format and verifies", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    module = MLIR.Module.create!(@scalar_module, ctx: ctx) |> MLIR.verify!()

    {_, op_names} =
      module
      |> Beaver.Walker.postwalk([], fn
        %MLIR.Operation{} = op, acc -> {op, [MLIR.Operation.name(op) | acc]}
        element, acc -> {element, acc}
      end)

    assert Enum.sort(op_names) ==
             Enum.sort([
               "builtin.module",
               "ex.func",
               "ex.lit",
               "ex.var",
               "ex.bind",
               "ex.add",
               "ex.call",
               "ex.return"
             ])
  end

  test "constructs ops with the Beaver DSL", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    module =
      mlir ctx: ctx do
        module do
          Ex.func sym_name: Attribute.string("main") do
            region do
              block do
                value =
                  Ex.lit(value: Attribute.integer(Type.i64(), 42)) >>>
                    Type.i64()

                Ex.return(value) >>> []
              end
            end
          end >>> []
        end
      end
      |> MLIR.verify!()

    assert MLIR.to_string(module, generic: true) =~ ~s{"ex.lit"}
    assert MLIR.to_string(module, generic: true) =~ ~s{"ex.return"}
  end

  test "constructs control flow ops with the Beaver DSL", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    module =
      mlir ctx: ctx do
        module do
          Ex.func sym_name: Attribute.string("main") do
            region do
              block do
                one =
                  Ex.lit(value: Attribute.integer(Type.i64(), 1)) >>>
                    Type.i64()

                two =
                  Ex.lit(value: Attribute.integer(Type.i64(), 2)) >>>
                    Type.i64()

                cond =
                  Ex.cmp(left: one, right: two, predicate: Attribute.string("slt")) >>>
                    Type.i64()

                result =
                  Ex.if cond: cond, operandSegmentSizes: :infer do
                    region do
                      block do
                        Ex.yield(values: one, operandSegmentSizes: :infer) >>> []
                      end
                    end

                    region do
                      block do
                        Ex.yield(values: two, operandSegmentSizes: :infer) >>> []
                      end
                    end
                  end >>>
                    Type.i64()

                Ex.return(result) >>> []
              end
            end
          end >>> []
        end
      end
      |> MLIR.verify!()

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ ~s{"ex.cmp"}
    assert rendered =~ ~s{"ex.if"}
    assert rendered =~ ~s{"ex.yield"}
  end

  test "constructs case/clause ops with the Beaver DSL", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    module =
      mlir ctx: ctx do
        module do
          Ex.func sym_name: Attribute.string("main") do
            region do
              block do
                scrutinee =
                  Ex.lit(value: Attribute.integer(Type.i64(), 1)) >>>
                    Type.i64()

                result =
                  Ex.case scrutinee: scrutinee, operandSegmentSizes: :infer do
                    region do
                      block do
                        Ex.clause(patterns: Attribute.array([Attribute.integer(Type.i64(), 1)])) >>>
                          []

                        ten =
                          Ex.lit(value: Attribute.integer(Type.i64(), 10)) >>>
                            Type.i64()

                        Ex.yield(values: ten, operandSegmentSizes: :infer) >>> []
                      end

                      block do
                        Ex.clause(patterns: Attribute.array([Attribute.integer(Type.i64(), 2)])) >>>
                          []

                        twenty =
                          Ex.lit(value: Attribute.integer(Type.i64(), 20)) >>>
                            Type.i64()

                        Ex.yield(values: twenty, operandSegmentSizes: :infer) >>> []
                      end
                    end
                  end >>>
                    Type.i64()

                Ex.return(result) >>> []
              end
            end
          end >>> []
        end
      end
      |> MLIR.verify!()

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ ~s{"ex.case"}
    assert rendered =~ ~s{"ex.clause"}
  end

  test "constructs term ops with the Beaver DSL", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    module =
      mlir ctx: ctx do
        module do
          Ex.func sym_name: Attribute.string("main") do
            region do
              block do
                one =
                  Ex.lit(value: Attribute.integer(Type.i64(), 1)) >>>
                    Type.i64()

                two =
                  Ex.lit(value: Attribute.integer(Type.i64(), 2)) >>>
                    Type.i64()

                one_boxed =
                  Ex.box(value: one) >>>
                    Ex.term()

                two_boxed =
                  Ex.box(value: two) >>>
                    Ex.term()

                tuple =
                  Ex.tuple(elements: [one_boxed, two_boxed], operandSegmentSizes: :infer) >>>
                    Ex.term()

                list =
                  Ex.list(elements: [one_boxed, two_boxed], operandSegmentSizes: :infer) >>>
                    Ex.term()

                map =
                  Ex.map(entries: [one_boxed, two_boxed], operandSegmentSizes: :infer) >>>
                    Ex.term()

                bin =
                  Ex.binary(segments: [one_boxed], operandSegmentSizes: :infer) >>>
                    Ex.term()

                _is_tuple =
                  Ex.is_tuple(value: tuple) >>>
                    Type.i64()

                _is_list =
                  Ex.is_list(value: list) >>>
                    Type.i64()

                _is_map =
                  Ex.is_map(value: map) >>>
                    Type.i64()

                _is_binary =
                  Ex.is_binary(value: bin) >>>
                    Type.i64()

                Ex.return(
                  Ex.is_integer(value: one) >>>
                    Type.i64()
                ) >>>
                  []
              end
            end
          end >>> []
        end
      end
      |> MLIR.verify!()

    rendered = MLIR.to_string(module, generic: true)
    assert rendered =~ ~s{"ex.box"}
    assert rendered =~ ~s{"ex.tuple"}
    assert rendered =~ ~s{"ex.list"}
    assert rendered =~ ~s{"ex.map"}
    assert rendered =~ ~s{"ex.binary"}
    assert rendered =~ ~s{"ex.is_tuple"}
    assert rendered =~ ~s{"ex.is_list"}
    assert rendered =~ ~s{"ex.is_map"}
    assert rendered =~ ~s{"ex.is_binary"}
    assert rendered =~ ~s{"ex.is_integer"}
  end

  test "rejects invalid attributes and terminator placement", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    assert_raise ArgumentError, ~r/expected base attribute/, fn ->
      ~S"""
      module {
        "ex.func"() ({
        ^bb0:
          %0 = "ex.lit"() {value = "not an integer"} : () -> i64
          "ex.return"(%0) {operandSegmentSizes = array<i32: 1>} : (i64) -> ()
        }) {sym_name = "bad"} : () -> ()
      }
      """
      |> MLIR.Module.create!(ctx: ctx)
    end

    assert_raise ArgumentError, ~r/must be the last operation in the parent block/, fn ->
      ~S"""
      module {
        "ex.func"() ({
        ^bb0:
          "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
          "ex.return"() {operandSegmentSizes = array<i32: 0>} : () -> ()
        }) {sym_name = "bad"} : () -> ()
      }
      """
      |> MLIR.Module.create!(ctx: ctx)
    end
  end

  test "round-trips the scalar subset through bytecode", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    original = MLIR.Module.create!(@scalar_module, ctx: ctx) |> MLIR.verify!()

    bytecode = MLIR.Bytecode.write!(original)
    fresh_ctx = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(fresh_ctx) end)

    assert Ex
           |> then(&Beaver.Slang.load(fresh_ctx, &1))
           |> MLIR.LogicalResult.success?()

    bytecode_roundtrip =
      bytecode
      |> MLIR.Bytecode.read!(ctx: fresh_ctx)
      |> MLIR.verify!()

    assert MLIR.to_string(bytecode_roundtrip, generic: true) ==
             MLIR.to_string(original, generic: true)
  end

  test "round-trips control flow ops through bytecode", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    original = MLIR.Module.create!(@control_flow_module, ctx: ctx) |> MLIR.verify!()

    bytecode = MLIR.Bytecode.write!(original)
    fresh_ctx = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(fresh_ctx) end)

    assert Ex
           |> then(&Beaver.Slang.load(fresh_ctx, &1))
           |> MLIR.LogicalResult.success?()

    bytecode_roundtrip =
      bytecode
      |> MLIR.Bytecode.read!(ctx: fresh_ctx)
      |> MLIR.verify!()

    assert MLIR.to_string(bytecode_roundtrip, generic: true) ==
             MLIR.to_string(original, generic: true)
  end

  test "round-trips case/clause ops through bytecode", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    original = MLIR.Module.create!(@case_module, ctx: ctx) |> MLIR.verify!()

    bytecode = MLIR.Bytecode.write!(original)
    fresh_ctx = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(fresh_ctx) end)

    assert Ex
           |> then(&Beaver.Slang.load(fresh_ctx, &1))
           |> MLIR.LogicalResult.success?()

    bytecode_roundtrip =
      bytecode
      |> MLIR.Bytecode.read!(ctx: fresh_ctx)
      |> MLIR.verify!()

    assert MLIR.to_string(bytecode_roundtrip, generic: true) ==
             MLIR.to_string(original, generic: true)
  end

  test "round-trips term ops through bytecode", %{ctx: ctx} do
    Beaver.Slang.load(ctx, Ex)

    original = MLIR.Module.create!(@term_module, ctx: ctx) |> MLIR.verify!()

    bytecode = MLIR.Bytecode.write!(original)
    fresh_ctx = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(fresh_ctx) end)

    assert Ex
           |> then(&Beaver.Slang.load(fresh_ctx, &1))
           |> MLIR.LogicalResult.success?()

    bytecode_roundtrip =
      bytecode
      |> MLIR.Bytecode.read!(ctx: fresh_ctx)
      |> MLIR.verify!()

    assert MLIR.to_string(bytecode_roundtrip, generic: true) ==
             MLIR.to_string(original, generic: true)
  end
end
