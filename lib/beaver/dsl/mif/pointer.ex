defmodule Beaver.MIF.Pointer do
  use Beaver.MIF.Intrinsic
  alias Beaver.MLIR.{Type, Attribute}
  alias Beaver.MLIR.Dialect.{Arith, LLVM}

  defi allocate(elem_type, opts) do
    handle_intrinsic(:allocate, [elem_type, 1], opts)
  end

  defi allocate(elem_type, size = %MLIR.Value{}, opts) do
    LLVM.alloca(size, elem_type: elem_type) >>> ~t{!llvm.ptr}
  end

  # when is_integer(size)
  defi allocate(elem_type, size, opts) when is_integer(size) do
    one = Arith.constant(value: Attribute.integer(Type.i(32), size)) >>> ~t<i32>
    handle_intrinsic(:allocate, [elem_type, one], opts)
  end

  defi load(type, ptr, opts) do
    LLVM.load(ptr) >>> type
  end

  defi store(val, ptr, opts) do
    LLVM.store(val, ptr) >>> []
  end

  defi element_ptr(elem_type, ptr, n, opts) do
    LLVM.getelementptr(ptr, n,
      elem_type: elem_type,
      rawConstantIndices: ~a{array<i32: -2147483648>}
    ) >>> ~t{!llvm.ptr}
  end

  defi t(opts) do
    Beaver.Deferred.from_opts(opts, ~t{!llvm.ptr})
  end
end
