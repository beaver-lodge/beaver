defmodule E2ETest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  describe "e2e compilation and JIT invocation" do
    import Beaver.Sigils
    import MLIR.{Transform, Conversion}

    def make_jit(ctx) do
      ~m"""
      module {
        func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
          %res = arith.addi %arg0, %arg1 : i32
          return %res : i32
        }
      }
      """.(ctx)
      |> canonicalize
      |> cse
      |> convert_arith_to_llvm()
      |> convert_func_to_llvm
      |> convert_arith_to_llvm
      |> Beaver.Composer.run!()
      |> MLIR.ExecutionEngine.create!()
    end

    test "fail to create JIT engine", %{ctx: ctx} do
      assert_raise ArgumentError,
                   ~r{Execution engine creation failed\nat -:2:3: .+for op: func.func\n  at -:2:3: see current operation: \n    \"func.func\".+\n    .+\n    },
                   fn ->
                     ~m"""
                     module {
                       func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
                         %res = arith.addi %arg0, %arg1 : i32
                         return %res : i32
                       }
                     }
                     """.(ctx)
                     |> MLIR.ExecutionEngine.create!()
                   end
    end

    test "dirty scheduler invocation", %{ctx: ctx} do
      arg = Beaver.Native.I32.make(42)
      return = Beaver.Native.I32.make(-1)
      jit = make_jit(ctx)
      MLIR.ExecutionEngine.invoke!(jit, "add", [arg, arg], return)
      assert return |> Beaver.Native.to_term() == 84
      MLIR.ExecutionEngine.invoke!(jit, "add", [return, arg], return, dirty: :cpu_bound)
      assert return |> Beaver.Native.to_term() == 126
      MLIR.ExecutionEngine.invoke!(jit, "add", [arg, return], return, dirty: :io_bound)
      assert return |> Beaver.Native.to_term() == 168
    end

    test "parallel invocation", %{ctx: ctx} do
      jit = make_jit(ctx)

      for i <- 0..100_0 do
        Task.async(fn ->
          arg = Beaver.Native.I32.make(i)
          return = Beaver.Native.I32.make(-1)
          return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg, arg], return)
          assert return |> Beaver.Native.to_term() == i * 2
        end)
      end
      |> Task.await_many()
    end
  end
end
