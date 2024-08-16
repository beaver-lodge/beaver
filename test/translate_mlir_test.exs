defmodule TranslateMLIRTest do
  use Beaver.Case, async: true

  def compare_with_original(func, args) do
    assert apply(AddTwoIntegers, func, args) ==
             apply(AddTwoIntegers, String.to_atom("__original__#{func}"), args)
  end

  @moduletag :smoke
  test "add two integers" do
    assert 3 == AddTwoIntegers.llvm_add(1, 2)
    assert 5 == AddTwoIntegers.llvm_add1(2, 2)
    assert 7 == AddTwoIntegers.llvm_add_multi_lines(1, 1)
    compare_with_original(:llvm_for_loop, [111, 111])
    compare_with_original(:llvm_add_multi_lines, [-300, 1])
    compare_with_original(:llvm_for_loop2, [2, 3])
    compare_with_original(:llvm_for_loop2, [2, -777])
  end
end
