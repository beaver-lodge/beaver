defmodule TestVariadic do
  @moduledoc "An example to showcase the a variadic dialect in IRDL test. Original MLIR file: https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/IRDL/variadics.irdl.mlir"
  use Beaver.Slang, name: "testvar"

  defop single_operand(i = {:single, Type.i32()}), do: []

  defop var_operand(a = Type.i16(), b = {:variadic, Type.i32()}, c = Type.i64()), do: []
  defop var_operand_alt(a = Type.i16(), b = {:variadic, a}, Type.i64()), do: []
  defop var_operand_alt1(a = Type.i16(), c = b = {:variadic, a}, b, Type.i64()), do: c
end
