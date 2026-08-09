defmodule LLVMDialectTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.LLVM

  require LLVM

  @moduletag :smoke

  test "builds contextual LLVM ABI types", %{ctx: ctx} do
    assert to_string(LLVM.pointer(ctx: ctx)) == "!llvm.ptr"
    assert to_string(LLVM.pointer(3, ctx: ctx)) == "!llvm.ptr<3>"
    assert to_string(LLVM.array(Type.i8(), 4, ctx: ctx)) == "!llvm.array<4 x i8>"

    assert to_string(LLVM.struct([Type.i32(), Type.f64()], packed: true, ctx: ctx)) ==
             "!llvm.struct<packed (i32, f64)>"

    assert to_string(LLVM.function_type([Type.i32()], [Type.i32()], ctx: ctx)) ==
             "!llvm.func<i32 (i32)>"

    assert to_string(LLVM.function_type([Type.i32()], [], vararg: true, ctx: ctx)) ==
             "!llvm.func<void (i32, ...)>"
  end

  test "builds validated ABI enum attributes", %{ctx: ctx} do
    assert to_string(LLVM.linkage(:internal, ctx: ctx)) == "#llvm.linkage<internal>"
    assert to_string(LLVM.calling_convention(:fast, ctx: ctx)) == "#llvm.cconv<fastcc>"

    assert_raise ArgumentError, ~r/unsupported LLVM linkage/, fn ->
      apply(LLVM, :linkage, [:local, [ctx: ctx]])
    end
  end

  test "defines, calls, and verifies an LLVM function", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          LLVM.func identity(
                      function_type: LLVM.function_type([Type.i32()], [Type.i32()]),
                      linkage: LLVM.linkage(:internal),
                      CConv: LLVM.calling_convention(:c)
                    ) do
            region do
              block _entry(arg >>> Type.i32()) do
                LLVM.return(arg) >>> []
              end
            end
          end

          LLVM.func caller(function_type: LLVM.function_type([], [Type.i32()])) do
            region do
              block do
                value = LLVM.mlir_constant(value: Attribute.integer(Type.i32(), 7)) >>> Type.i32()
                result = LLVM.call_(value, callee: :identity) >>> Type.i32()
                LLVM.return(result) >>> []
              end
            end
          end
        end
      end

    MLIR.verify!(module)
    assert to_string(module) =~ "llvm.func internal @identity"
    assert to_string(module) =~ "llvm.call @identity"
  end

  test "builds an inline initialized global", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          LLVM.global(
            sym_name: :answer,
            type: Type.i32(),
            value: Attribute.integer(Type.i32(), 42),
            constant: true,
            linkage: :internal,
            alignment: 4
          ) >>> []
        end
      end

    MLIR.verify!(module)
    assert to_string(module) =~ "llvm.mlir.global internal constant @answer"
    assert to_string(module) =~ "alignment = 4"
  end

  test "builds compatible DICompileUnit attributes with an optional source language dialect", %{
    ctx: ctx
  } do
    legacy =
      LLVM.di_compile_unit(
        ctx: ctx,
        source_language: :c,
        filename: "legacy.c",
        producer: "Beaver"
      )

    assert to_string(legacy) =~ "sourceLanguage = DW_LANG_C"
    refute to_string(legacy) =~ "sourceLanguageDialect"

    dialect_opts = [
      ctx: ctx,
      source_language: :c,
      source_language_dialect: :tile,
      filename: "kernel.c",
      producer: "Beaver"
    ]

    if LLVM.di_compile_unit_source_language_dialect_supported?() do
      dialect = LLVM.di_compile_unit(dialect_opts)
      assert to_string(dialect) =~ "sourceLanguageDialect = DW_LLVM_LANG_DIALECT_tile"
    else
      assert_raise ArgumentError, ~r/does not support DICompileUnit/, fn ->
        LLVM.di_compile_unit(dialect_opts)
      end
    end
  end
end
