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
    end

    assert 3 == AddTwoInt.llvm_add(1, 2)
    assert 3 == AddTwoInt.llvm_add1(1, 1)
  end
end
