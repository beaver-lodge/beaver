defmodule SlangCompleteTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR

  @moduletag :smoke

  @valid_module ~S"""
  module {
    "complete_slang.scope"() ({
    ^bb0:
    }) {label = "ok"} : () -> ()
  }
  """

  test "emits a complete, named IRDL schema", %{ctx: ctx} do
    schema = CompleteSlang.__slang_dialect__(ctx) |> MLIR.verify!()
    rendered = MLIR.to_string(schema, generic: true)

    assert rendered =~ ~s{"irdl.all_of"}
    assert rendered =~ ~s{base_name = "!builtin.integer"}
    assert rendered =~ ~s{base_name = "#builtin.string"}
    assert rendered =~ ~s{base_ref = @complete_slang::@token}
    assert rendered =~ ~s{names = ["element"]}
    assert rendered =~ ~s{names = ["item"]}
    assert rendered =~ ~s{names = ["value"]}
    assert rendered =~ ~s{names = ["head", "tail", "fallback"]}
    assert rendered =~ ~s{names = ["input"]}
    assert rendered =~ ~s{names = ["output"]}
    assert rendered =~ ~s{variadicity_array[single, variadic, optional]}
    assert rendered =~ ~s{variadicity_array[optional]}
    assert rendered =~ ~s{"irdl.attributes"}
    assert rendered =~ ~s{attributeValueNames = ["label"]}
    assert rendered =~ ~s{"irdl.regions"}
    assert rendered =~ ~s{names = ["body"]}
  end

  test "loads types, attributes, operation attributes, regions, and dynamic traits", %{ctx: ctx} do
    assert CompleteSlang.__slang_traits__() == [
             {"scope", [:isolated_from_above, :no_terminator]},
             {"yield", [:terminator]}
           ]

    assert CompleteSlang |> then(&Beaver.Slang.load(ctx, &1)) |> MLIR.LogicalResult.success?()
    assert MLIR.Context.terminator?(ctx, "complete_slang.yield")

    type = CompleteSlang.box(MLIR.Type.i32()) |> Beaver.Deferred.resolve(ctx)

    attribute =
      CompleteSlang.direction(MLIR.Attribute.string("left")) |> Beaver.Deferred.resolve(ctx)

    refute MLIR.null?(type)
    refute MLIR.null?(attribute)
    assert MLIR.to_string(type) == "!complete_slang.box<i32>"
    assert MLIR.to_string(attribute) == ~s{#complete_slang.direction<"left">}

    assert @valid_module |> MLIR.Module.create!(ctx: ctx) |> MLIR.verify?()

    assert_raise ArgumentError, ~r/expected base attribute 'builtin.string'/, fn ->
      ~S[module { "complete_slang.scope"() ({ ^bb0: }) {label = 42 : i32} : () -> () }]
      |> MLIR.Module.create!(ctx: ctx)
    end
  end

  test "enforces isolated-from-above and terminator semantics", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CompleteSlang)

    assert_raise ArgumentError, ~r/using value defined outside the region/, fn ->
      ~S"""
      module {
        func.func @bad(%arg0: i32) {
          "complete_slang.scope"() ({
          ^bb0:
            "complete_slang.consume"(%arg0) : (i32) -> ()
          }) {label = "bad"} : () -> ()
          return
        }
      }
      """
      |> MLIR.Module.create!(ctx: ctx)
    end

    assert_raise ArgumentError, ~r/must be the last operation in the parent block/, fn ->
      ~S"""
      module {
        "complete_slang.scope"() ({
        ^bb0:
          "complete_slang.yield"() : () -> ()
          "complete_slang.yield"() : () -> ()
        }) {label = "bad"} : () -> ()
      }
      """
      |> MLIR.Module.create!(ctx: ctx)
    end
  end

  test "round-trips a dynamic dialect through text and bytecode", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CompleteSlang)
    original = MLIR.Module.create!(@valid_module, ctx: ctx) |> MLIR.verify!()

    text_roundtrip =
      original
      |> MLIR.to_string(generic: true)
      |> MLIR.Module.create!(ctx: ctx)
      |> MLIR.verify!()

    bytecode = MLIR.Bytecode.write!(text_roundtrip)
    fresh_ctx = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(fresh_ctx) end)

    refute MLIR.Context.terminator?(fresh_ctx, "complete_slang.yield")

    assert CompleteSlang
           |> then(&Beaver.Slang.load(fresh_ctx, &1))
           |> MLIR.LogicalResult.success?()

    assert MLIR.Context.terminator?(fresh_ctx, "complete_slang.yield")

    bytecode_roundtrip =
      bytecode
      |> MLIR.Bytecode.read!(ctx: fresh_ctx)
      |> MLIR.verify!()

    assert MLIR.to_string(bytecode_roundtrip, generic: true) ==
             MLIR.to_string(original, generic: true)
  end

  test "reports invalid constraints at their Slang declaration", %{ctx: ctx} do
    Code.compile_string(
      ~S"""
      defmodule InvalidSlangSourceLocation do
        use Beaver.Slang, name: "invalid_slang_source_location"
        deftype broken(value = base("invalid"))
      end
      """,
      "/tmp/invalid_slang_source_location.ex"
    )

    {:error, diagnostics} =
      apply(InvalidSlangSourceLocation, :__slang_dialect__, [ctx])
      |> MLIR.verify()

    message = MLIR.Diagnostic.format(diagnostics)
    assert message =~ "/tmp/invalid_slang_source_location.ex:3"
    assert message =~ "the base type or attribute name should start with '!' or '#'"
  end
end
