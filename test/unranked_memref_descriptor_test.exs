defmodule TestingUnrankedMemRefABI do
  @moduledoc false
  use Beaver
  alias MLIR.Dialect.{Func, Linalg, MemRef, Arith}
  alias MLIR.Type
  require Func
  use ENIFSupport

  @impl ENIFSupport
  def create(ctx) do
    elem_type = Type.i32(ctx: ctx)
    unranked_memref_t = Type.unranked_memref!(elem_type)

    mlir ctx: ctx do
      module do
        Func.func assign_meta(function_type: Type.function([], [unranked_memref_t])) do
          region do
            block _() do
              v = Arith.constant(value: Attribute.integer(elem_type, 100)) >>> ~t<i32>

              m =
                MemRef.alloc(operand_segment_sizes: :infer) >>>
                  Type.memref!([2, 3], elem_type)

              Linalg.fill inputs: v, outputs: m, operand_segment_sizes: :infer do
                region do
                  block _(in_arg >>> elem_type, _ >>> elem_type) do
                    Linalg.yield(in_arg) >>> []
                  end
                end
              end >>>
                []

              u = MemRef.cast(m) >>> unranked_memref_t

              Func.return(u) >>> []
            end
          end
        end
      end
    end
  end
end

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
    d = UnrankedMemRefDescriptor.empty(2)
    opaque_ptr = d |> UnrankedMemRefDescriptor.opaque_ptr()

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
