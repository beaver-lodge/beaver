defmodule DefineMLIRTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "cf with mutation", test_context do
    defmodule AddTwoInt do
      use TranslateMLIR

      defm llvm_add(a :: i64, b :: i64) do
        some_llvm_add_mlir_operation(a, b)
      end
    end

    AddTwoInt.llvm_add(1, 2)
  end
end
