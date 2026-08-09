defmodule WasmSSADialectTest do
  use Beaver
  use Beaver.Case, async: true

  require Beaver.MLIR.Dialect.WasmSSA

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.WasmSSA
  alias MLIR.Type

  test "wasmssa ops are bound from the registry" do
    assert function_exported?(WasmSSA, :func, 0)
    assert function_exported?(WasmSSA, :const, 0)
    assert function_exported?(WasmSSA, :return, 0)
    assert function_exported?(WasmSSA, :local_get, 0)
    assert function_exported?(WasmSSA, :call, 0)
    assert function_exported?(WasmSSA, :branch_if, 0)
  end

  test "parses wasmssa text", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        wasmssa.func @f() -> i32 {
          %0 = wasmssa.const 10 : i32
          wasmssa.return %0 : i32
        }
        """,
        ctx: ctx
      )

    text = MLIR.to_string(module)
    assert text =~ "wasmssa.func @f() -> i32"
    assert text =~ "wasmssa.const 10 : i32"
  end

  test "constructs wasmssa ops from the DSL", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          WasmSSA.func f(
                         functionType: Type.function([], [Type.i32()]),
                         sym_name: MLIR.Attribute.string("f")
                       ) do
            region do
              block _() do
                c = WasmSSA.const(value: MLIR.Attribute.integer(Type.i32(), 10)) >>> Type.i32()
                WasmSSA.return(c) >>> []
              end
            end
          end
        end
      end

    MLIR.verify!(module)
    text = MLIR.to_string(module)
    assert text =~ "wasmssa.func @f() -> i32"
    assert text =~ "wasmssa.const 10 : i32"
  end
end
