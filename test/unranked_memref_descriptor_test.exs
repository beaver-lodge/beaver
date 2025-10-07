defmodule UnrankedMemRefDescriptorTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR.UnrankedMemRefDescriptor

  test "UnrankedMemRefDescriptor creation" do
    d = UnrankedMemRefDescriptor.empty(2)
    assert 2 = UnrankedMemRefDescriptor.rank(d)
  end

  test "UnrankedMemRefDescriptor with rank 0" do
    d = UnrankedMemRefDescriptor.empty()
    assert 0 = UnrankedMemRefDescriptor.rank(d)
  end

  test "UnrankedMemRefDescriptor with rank 3" do
    d = UnrankedMemRefDescriptor.empty(3)
    assert 3 = UnrankedMemRefDescriptor.rank(d)
  end

  test "abi", %{ctx: ctx} do
    d = UnrankedMemRefDescriptor.empty()
    opaque_ptr = d |> Beaver.Native.opaque_ptr()

    TestingUnrankedMemRefABI.init(ctx)
    |> tap(fn %ENIFSupport{engine: e} ->
      MLIR.ExecutionEngine.invoke!(e, "assign_meta", [
        opaque_ptr
      ])
    end)

    assert 2 = UnrankedMemRefDescriptor.rank(d)
    assert 0 = UnrankedMemRefDescriptor.offset(d)
    assert [2, 3] = UnrankedMemRefDescriptor.sizes(d)
    assert [3, 1] = UnrankedMemRefDescriptor.strides(d)
    assert :ok = UnrankedMemRefDescriptor.free(d)
    assert :noop = UnrankedMemRefDescriptor.free(d)
  end
end
