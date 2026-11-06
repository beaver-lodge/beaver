defmodule LLVMDialectTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Type}
  alias Beaver.MLIR.Dialect.LLVM

  require LLVM

  @moduletag :smoke

  test "defines, calls, and verifies an LLVM function", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          LLVM.func identity(
                      function_type: Type.get("!llvm.func<i32 (i32)>"),
                      linkage: MLIR.Attribute.get("#llvm.linkage<internal>"),
                      CConv: MLIR.Attribute.get("#llvm.cconv<ccc>")
                    ) do
            region do
              block _entry(arg >>> Type.i32()) do
                LLVM.return(arg) >>> []
              end
            end
          end

          LLVM.func caller(function_type: Type.get("!llvm.func<i32 ()>")) do
            region do
              block do
                value = LLVM.mlir_constant(value: Attribute.integer(Type.i32(), 7)) >>> Type.i32()

                result =
                  LLVM.call(
                    callee_operands: [value],
                    callee: MLIR.Attribute.flat_symbol_ref("identity"),
                    operand_segment_sizes: :infer,
                    op_bundle_sizes: MLIR.Attribute.get("array<i32>")
                  ) >>> Type.i32()

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
end
