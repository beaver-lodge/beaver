defmodule TranslateMLIRTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "add two integers", test_context do
    defmodule AddTwoInt do
      use TranslateMLIR

      mlir_func llvm_add(a :: i64, b :: i64) do
        a + b
      end

      mlir_func llvm_add1(a :: i64, b :: i64) do
        a + b + 1
      end

      mlir_func llvm_add_multi_lines(a :: i64, b :: i64) do
        c = a
        c = c + b + 1
        a = 2 + c + b + 1
        a
      end

      mlir_func llvm_for_loop(a :: i64, b :: i64) do
        for i <- [1, 2, 3, 4, 5] do
          i + a + b
        end
      end

      mlir_func llvm_for_loop2(a :: i64, b :: i64) do
        l =
          for i <- [1, 2, 3] do
            i + a + b
          end

        m =
          for i <- [1, 2, 3, 4] do
            i + a + b
          end

        for i <- l, j <- m, k <- m do
          i + j + k + a + b
        end
      end

      def compare_with_original(func, args) do
        assert apply(__MODULE__, func, args) ==
                 apply(__MODULE__, String.to_atom("__original__#{func}"), args)
      end
    end

    assert 3 == AddTwoInt.llvm_add(1, 2)
    assert 5 == AddTwoInt.llvm_add1(2, 2)
    assert 7 == AddTwoInt.llvm_add_multi_lines(1, 1)
    AddTwoInt.compare_with_original(:llvm_for_loop, [111, 111])
    AddTwoInt.compare_with_original(:llvm_add_multi_lines, [-300, 1])
    AddTwoInt.compare_with_original(:llvm_for_loop2, [2, 3])
    AddTwoInt.compare_with_original(:llvm_for_loop2, [2, -777])
  end
end
