defmodule MIFTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "add two integers", test_context do
    defmodule AddTwoInt do
      use Beaver.MIF

      defm add(a :: i64(), b :: i64()) do
        op llvm.add(a, b) :: i64()
      end

      defm subtract(a :: i32(), b :: i32()) do
        op llvm.sub(a, b) :: i32()
      end
    end

    Beaver.MIF.init_jit(AddTwoInt)
    assert AddTwoInt.add(1, 2) == 3
    assert AddTwoInt.subtract(30, 20) == 10
    assert AddTwoInt.subtract(30, "") == -1111
    Beaver.MIF.destroy_jit(AddTwoInt)
  end
end
