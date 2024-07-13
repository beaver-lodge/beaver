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
    s_table = op |> MLIR.Operation.from_module() |> MLIR.CAPI.mlirSymbolTableCreate()
    found = MLIR.CAPI.mlirSymbolTableLookup(s_table, MLIR.StringRef.create("enif_make_int64"))

    if MLIR.is_null(found) do
      raise "Function not found"
    end

    if not MLIR.Dialect.Func.is_external(found) do
      raise "Function is not external"
    end

    MLIR.CAPI.mlirSymbolTableDestroy(s_table)
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
      end
    end
  end
end
