defmodule Beaver.MLIR.Target.LLVMIRTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Target.LLVMIR

  @moduletag :smoke

  test "exports an LLVM dialect module as LLVM IR text", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        module {
          llvm.func @answer() -> i32 {
            %answer = llvm.mlir.constant(42 : i32) : i32
            llvm.return %answer : i32
          }
        }
        """,
        ctx: ctx
      )

    assert {:ok, llvm_ir} = LLVMIR.translate(module)
    assert llvm_ir =~ "define i32 @answer()"
    assert llvm_ir =~ "ret i32 42"
    assert LLVMIR.translate!(module) == llvm_ir
  end

  test "returns diagnostics for a module that is not lowered to LLVM dialect", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        module {
          func.func @answer() -> i32 {
            %answer = arith.constant 42 : i32
            return %answer : i32
          }
        }
        """,
        ctx: ctx
      )

    assert {:error, diagnostics} = LLVMIR.translate(module)
    assert diagnostics != []

    assert_raise ArgumentError, ~r/failed to translate module to LLVM IR/, fn ->
      LLVMIR.translate!(module)
    end
  end

  test "can repeatedly translate the same module", %{ctx: ctx} do
    module = MLIR.Module.create!("module { llvm.func @noop() { llvm.return } }", ctx: ctx)

    for _ <- 1..20 do
      assert {:ok, llvm_ir} = LLVMIR.translate(module)
      assert llvm_ir =~ "define void @noop()"
    end
  end
end
