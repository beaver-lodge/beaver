defmodule E2ETest do
  use ExUnit.Case
  alias Beaver.MLIR

  test "run mlir module defined by sigil" do
    import Beaver.MLIR.Sigils
    import MLIR.{Transforms, Conversion}

    arg = Exotic.Value.get(42)
    return = Exotic.Value.get(-1)

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

    assert return |> Exotic.Value.extract() == 84

    for i <- 0..100_0 do
      Task.async(fn ->
        arg = Exotic.Value.get(i)
        return = Exotic.Value.get(-1)
        return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg, arg], return)
        # return here is a resource reference
        assert return == return
        assert return |> Exotic.Value.extract() == i * 2
      end)
    end
    |> Task.await_many()

    # |> llvm

    ~t<i64>
    ~a/(i64, i64) -> (i64)/
  end
end
