defmodule AddENIF do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias Beaver.MLIR
  alias MLIR.Type
  alias MLIR.Dialect.{Func, Arith, Ptr, MemRef}
  require Func
  use ENIFSupport

  @impl ENIFSupport
  def after_verification(op) do
    op
    |> MLIR.Operation.from_module()
    |> MLIR.Operation.with_symbol_table(fn s_table ->
      found = MLIR.CAPI.mlirSymbolTableLookup(s_table, MLIR.StringRef.create("enif_make_int64"))

      if MLIR.null?(found) do
        raise "Function not found"
      end

      if not MLIR.Dialect.Func.external?(found) do
        raise "Function is not external"
      end
    end)

    op
  end

  @impl ENIFSupport
  def create(ctx) do
    mlir ctx: ctx do
      module do
        Beaver.ENIF.declare_external_functions(Beaver.Env.context(), Beaver.Env.block())
        env_t = ENIF.Type.env()
        term_t = ENIF.Type.term()

        Func.func add(function_type: Type.function([env_t, term_t, term_t], [term_t])) do
          region do
            block _(env >>> env_t, left >>> term_t, right >>> term_t) do
              left_ptr =
                MemRef.alloca(operand_segment_sizes: :infer) >>>
                  Type.memref!([], Type.i64(ctx: ctx), memory_space: ~a{#ptr.generic_space})

              right_ptr =
                MemRef.alloca(operand_segment_sizes: :infer) >>>
                  Type.memref!([], Type.i64(ctx: ctx), memory_space: ~a{#ptr.generic_space})

              left_ptr_arg = Ptr.to_ptr(left_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}
              right_ptr_arg = Ptr.to_ptr(right_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}
              ENIF.get_int64(env, left, left_ptr_arg) >>> :infer
              ENIF.get_int64(env, right, right_ptr_arg) >>> :infer
              left = MemRef.load(left_ptr) >>> Type.i64()
              right = MemRef.load(right_ptr) >>> Type.i64()
              sum = Arith.addi(left, right) >>> Type.i64()
              sum = ENIF.make_int64(env, sum) >>> :infer
              Func.return(sum) >>> []
            end
          end
        end

        Func.func add_point_one(function_type: Type.function([env_t], [term_t])) do
          region do
            block _(env >>> env_t) do
              f = Attribute.float(Type.f64(), 0.1)
              f = Arith.constant(value: f) >>> ~t<f64>
              f = ENIF.make_double(env, f) >>> :infer
              Func.return(f) >>> []
            end
          end
        end
      end
    end
  end
end
