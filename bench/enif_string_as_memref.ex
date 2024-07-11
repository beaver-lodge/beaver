defmodule ENIFStringAsMemRef do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias Beaver.MLIR
  alias MLIR.{Attribute, Type}
  alias MLIR.Dialect.{Func, MemRef, Index, LLVM, Arith}
  require Func
  use ENIFSupport

  @impl ENIFSupport
  def create(ctx) do
    mlir ctx: ctx do
      module do
        Beaver.ENIF.populate_external_functions(Beaver.Env.context(), Beaver.Env.block())
        env_t = ENIF.Type.env()
        term_t = ENIF.Type.term()

        Func.func original_str(function_type: Type.function([env_t, term_t], [term_t])) do
          region do
            block _(_env >>> env_t, s >>> term_t) do
              Func.return(s) >>> []
            end
          end
        end

        Func.func str_as_memref_get_len(function_type: Type.function([env_t, term_t], [term_t])) do
          region do
            block _(env >>> env_t, s >>> term_t) do
              f = :inspect_binary_as_memref
              {[t0, t1], memref_t} = Beaver.ENIF.signature(Beaver.Env.context(), f)
              symbol = Attribute.flat_symbol_ref(f)
              unless MLIR.Type.equal?(t0, env_t), do: raise("Expected equal types")
              unless MLIR.Type.equal?(t1, term_t), do: raise("Expected equal types")
              m = Func.call(env, s, callee: symbol) >>> memref_t
              zero = Index.constant(value: Attribute.index(0)) >>> Type.index()
              one = Arith.constant(value: Attribute.integer(Type.i(32), 1)) >>> ~t<i32>
              len = MemRef.dim(m, zero) >>> :infer
              len = Index.casts(len) >>> ~t<i64>
              len_ptr = LLVM.alloca(one, elem_type: Type.i32()) >>> ~t{!llvm.ptr}
              LLVM.store(len, len_ptr) >>> []
              maker = :enif_make_int64
              {_, t} = Beaver.ENIF.signature(Beaver.Env.context(), maker)
              symbol = Attribute.flat_symbol_ref(maker)
              len = Func.call(env, len, callee: symbol) >>> t
              Func.return(len) >>> []
            end
          end
        end
      end
    end
  end
end
