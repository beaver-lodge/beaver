defmodule AddENIF do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias Beaver.MLIR

  alias MLIR.Dialect.{Func, Arith, LLVM}
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
              size = Attribute.integer(Type.i32(), 64)
              size = Arith.constant(value: size) >>> ~t<i32>
              left_ptr = LLVM.alloca(size, elem_type: Type.i64()) >>> ~t{!llvm.ptr}
              right_ptr = LLVM.alloca(size, elem_type: Type.i64()) >>> ~t{!llvm.ptr}
              ENIF.get_int64(env, left, left_ptr) >>> :infer
              ENIF.get_int64(env, right, right_ptr) >>> :infer
              left = LLVM.load(left_ptr) >>> Type.i64()
              right = LLVM.load(right_ptr) >>> Type.i64()
              sum = Arith.addi(left, right) >>> Type.i64()
              sum = ENIF.make_int64(env, sum) >>> :infer
              Func.return(sum) >>> []
            end
          end
        end

        Func.func point_one(function_type: Type.function([env_t], [term_t])) do
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
