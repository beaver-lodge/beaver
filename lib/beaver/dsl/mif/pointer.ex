defmodule Beaver.MIF.Pointer do
  use Beaver
  alias Beaver.MLIR.{Type, Attribute}
  alias Beaver.MLIR.Dialect.{Arith, LLVM}

  def handle_intrinsic(:allocate, [elem_type], opts) do
    handle_intrinsic(:allocate, [elem_type, 1], opts)
  end

  def handle_intrinsic(:allocate, [elem_type, size], opts) when is_integer(size) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      one = Arith.constant(value: Attribute.integer(Type.i(32), size)) >>> ~t<i32>
      handle_intrinsic(:allocate, [elem_type, one], opts)
    end
  end

  def handle_intrinsic(:allocate, [elem_type, size = %MLIR.Value{}], opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      LLVM.alloca(size, elem_type: elem_type) >>> ~t{!llvm.ptr}
    end
  end

  def handle_intrinsic(:load, [type, ptr], opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      LLVM.load(ptr) >>> type
    end
  end

  def handle_intrinsic(:store, [val, ptr], opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      LLVM.store(val, ptr) >>> []
    end
  end

  def handle_intrinsic(:element_ptr, [elem_type, ptr, n], opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      LLVM.getelementptr(ptr, n,
        elem_type: elem_type,
        rawConstantIndices: ~a{array<i32: -2147483648>}
      ) >>> ~t{!llvm.ptr}
    end
  end

  def handle_intrinsic(:t, [], opts) do
    Beaver.Deferred.from_opts(opts, ~t{!llvm.ptr})
  end
end
