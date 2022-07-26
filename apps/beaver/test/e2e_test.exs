defmodule E2ETest do
  use ExUnit.Case
  alias Beaver.MLIR

  test "run mlir module defined by sigil" do
    import Beaver.MLIR.Sigils
    import MLIR.{Transforms, Conversion}

    arg = MLIR.CAPI.I32.make(42)
    return = MLIR.CAPI.I32.make(-1)

    jit =
      ~m"""
      module {
        func.func @add(%arg0 : i32, %arg1 : i32) -> i32 attributes { llvm.emit_c_interface } {
          %res = arith.addi %arg0, %arg1 : i32
          return %res : i32
        }
      }
      """
      |> canonicalize
      |> cse
      |> convert_func_to_llvm
      |> convert_arith_to_llvm
      |> MLIR.Pass.Composer.run!()
      |> MLIR.ExecutionEngine.create!()

    return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg, arg], return)

    assert return |> MLIR.CAPI.to_term() == 84

    for i <- 0..100_0 do
      Task.async(fn ->
        arg = MLIR.CAPI.I32.make(i)
        return = MLIR.CAPI.I32.make(-1)
        return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg, arg], return)
        # return here is a resource reference
        assert return == return
        assert return |> MLIR.CAPI.to_term() == i * 2
      end)
    end
    |> Task.await_many()
  end
end
