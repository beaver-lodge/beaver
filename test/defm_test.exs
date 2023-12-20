defmodule DefineMLIRTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "add two integers", test_context do
    defmodule AddTwoInt do
      use TranslateMLIR

      defm llvm_add(a :: i64, b :: i64) do
        a + b
      end

      defm llvm_add1(a :: i64, b :: i64) do
        a + b + 1
      end

      defm llvm_add_multi_lines(a :: i64, b :: i64) do
        c = a
        c = c + b + 1
        a = 2 + c + b + 1
        a
      end

      defm llvm_for_loop(a :: i64, b :: i64) do
        for i <- [1, 2, 3] do
          i + a + b
        end
      end
    end

    assert 3 == AddTwoInt.llvm_add(1, 2)
    assert 5 == AddTwoInt.llvm_add1(2, 2)
    assert 7 == AddTwoInt.llvm_add_multi_lines(1, 1)
  end
end
