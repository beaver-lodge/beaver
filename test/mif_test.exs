defmodule MIFTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "add two integers", test_context do
    defmodule AddTwoInt do
      use Beaver.MIF

      defm add(a :: i64(), b :: i64()) do
        op llvm.add(a, b) :: i64()
      end

      defm subtract(a :: i64(), b :: i64()) do
        op llvm.subtract(a, b) :: i64()
      end
    end

    AddTwoInt.add(1, 2) |> IO.puts()
  end
end
