defmodule CMath do
  @moduledoc "An example to showcase the cmath dialect in IRDL test. Original MLIR file: https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/IRDL/cmath.irdl.mlir"
  use Beaver.Slang, name: "cmath"

  defalias any_f do
    any_of([Type.f32(), Type.f64()])
  end

  deftype complex(t = ^any_f)
  defattr some_attr(t = ^any_f)

  defalias any_complex, do: complex(any())

  defalias any_complex2, do: complex(^any_f)

  defop norm(t = ^any_complex) do
    any()
  end

  defop mul(c = ^any_complex2, c) do
    c
  end
end
