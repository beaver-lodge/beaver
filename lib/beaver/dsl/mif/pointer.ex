defmodule Beaver.MIF.Pointer do
  use Beaver
  alias Beaver.MLIR.{Type, Attribute}
  alias Beaver.MLIR.Dialect.{Arith, LLVM}

  def handle_intrinsic(:allocate, [elem_type], opts) do
    handle_intrinsic(:allocate, [elem_type, 1], opts)
  end

  def handle_intrinsic(:allocate, [elem_type, size], opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      one = Arith.constant(value: Attribute.integer(Type.i(32), size)) >>> ~t<i32>
      LLVM.alloca(one, elem_type: elem_type) >>> ~t{!llvm.ptr}
    end
  end

  def handle_intrinsic(:load, [type, ptr], opts) do
    mlir ctx: opts[:ctx], block: opts[:block] do
      LLVM.load(ptr) >>> type
    end
  end
end
