defmodule TestVariadic do
  @moduledoc "An example to showcase the a variadic dialect in IRDL test. Original MLIR file: https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/IRDL/variadics.irdl.mlir"
  use Beaver.Slang, name: "testvar"

  defop single_operand(i = {:single, Type.i32()}), do: []
  defop var_operand(a = Type.i16(), b = {:variadic, Type.i32()}, c = Type.i64()), do: []
  defop var_operand_alt(a = Type.i16(), b = {:variadic, a}, Type.i64()), do: []
  defop var_operand_alt1(a = Type.i16(), c = b = {:variadic, a}, b, Type.i64()), do: c
  defop opt_operand(a = Type.i16(), b = {:optional, Type.i32()}, c = Type.i64()), do: []
  defop var_and_opt_operand({:variadic, Type.i16()}, {:optional, Type.i32()}, Type.i64()), do: []
  defop single_result(), do: [{:single, Type.i32()}]
  defop var_result(), do: [{:variadic, Type.i32()}]
  defop opt_result(), do: [{:optional, Type.i32()}]
  defop var_and_opt_result(), do: [{:variadic, Type.i16()}, {:optional, Type.i32()}, Type.i64()]

  defop var_and_opt_result_alt(x = is(Type.i16())) do
    a = is(Type.i16())
    b = is(Type.i32())
    c = is(Type.i64())
    [{:variadic, a}, {:optional, b}, c, c, x]
  end
end
