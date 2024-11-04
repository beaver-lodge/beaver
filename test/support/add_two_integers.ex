defmodule AddTwoIntegers do
  @moduledoc false
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
end
