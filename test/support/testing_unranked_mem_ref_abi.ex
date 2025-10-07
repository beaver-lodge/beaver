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
