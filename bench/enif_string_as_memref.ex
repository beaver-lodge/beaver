defmodule ENIFStringAsMemRef do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias MLIR.Dialect.{Func, MemRef, Index, LLVM, Arith, CF}
  require Func
  use ENIFSupport

  @impl ENIFSupport
  def create(ctx) do
    mlir ctx: ctx do
      module do
        Beaver.ENIF.declare_external_functions(Beaver.Env.context(), Beaver.Env.block())
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
              one = Arith.constant(value: Attribute.integer(Type.i(32), 1)) >>> :infer
              b_ptr = LLVM.alloca(one, elem_type: ENIF.Type.binary()) >>> ~t{!llvm.ptr}
              ENIF.inspect_binary(env, s, b_ptr) >>> :infer
              b = LLVM.load(b_ptr) >>> ENIF.Type.binary()
              size = LLVM.extractvalue(b, position: ~a{array<i64: 0>}) >>> Type.i64()
              d_ptr = LLVM.extractvalue(b, position: ~a{array<i64: 1>}) >>> ~t{!llvm.ptr}
              m = ENIF.ptr_to_memref(d_ptr, size) >>> :infer
              zero = Index.constant(value: Attribute.index(0)) >>> Type.index()
              len = MemRef.dim(m, zero) >>> :infer
              len = Index.casts(len) >>> ~t<i64>
              len = ENIF.make_int64(env, len) >>> :infer
              Func.return(len) >>> []
            end
          end
        end

        msg = "not a binary"
        global = MemRef.global(msg) >>> :infer
        global_t = MLIR.Attribute.unwrap(global[:type])
        global_s = MLIR.Attribute.unwrap(global[:sym_name])

        Func.func alloc_bin_and_copy(function_type: Type.function([env_t, term_t], [term_t])) do
          region do
            block _(env >>> env_t, s >>> term_t) do
              one = Arith.constant(value: Attribute.integer(Type.i(32), 1)) >>> :infer
              b_ptr = LLVM.alloca(one, elem_type: ENIF.Type.binary()) >>> ~t{!llvm.ptr}
              success = ENIF.inspect_binary(env, s, b_ptr) >>> :infer

              success =
                Arith.cmpi(success, one, predicate: Arith.cmp_i_predicate(:eq)) >>> Type.i1()

              CF.cond_br(success, Beaver.Env.block(successor), Beaver.Env.block(exit_blk)) >>> []
            end

            block exit_blk() do
              msg = MemRef.get_global(name: Attribute.flat_symbol_ref(global_s)) >>> global_t

              size =
                msg
                |> MLIR.Value.type()
                |> MLIR.CAPI.mlirShapedTypeGetDimSize(0)
                |> Beaver.Native.to_term()

              size = Arith.constant(value: Attribute.integer(Type.i(64), size)) >>> :infer
              term_ptr = LLVM.alloca(one, elem_type: ENIF.Type.term()) >>> ~t{!llvm.ptr}
              d_ptr = ENIF.make_new_binary(env, size, term_ptr) >>> :infer
              m = ENIF.ptr_to_memref(d_ptr, size) >>> :infer
              loc = m |> MLIR.Value.owner!() |> MLIR.Operation.location() |> MLIR.to_string()
              MemRef.copy(msg, m) >>> []
              msg = LLVM.load(term_ptr) >>> ENIF.Type.term()
              e = ENIF.raise_exception(env, msg) >>> []
              Func.return(e) >>> []
            end

            if !(loc =~ __ENV__.file) do
              raise "wrong location"
            end

            block successor() do
              b = LLVM.load(b_ptr) >>> ENIF.Type.binary()
              size = LLVM.extractvalue(b, position: ~a{array<i64: 0>}) >>> Type.i64()
              d_ptr = LLVM.extractvalue(b, position: ~a{array<i64: 1>}) >>> ~t{!llvm.ptr}
              m1 = ENIF.ptr_to_memref(d_ptr, size) >>> :infer
              term_ptr = LLVM.alloca(one, elem_type: ENIF.Type.term()) >>> ~t{!llvm.ptr}
              d_ptr = ENIF.make_new_binary(env, size, term_ptr) >>> :infer
              m2 = ENIF.ptr_to_memref(d_ptr, size) >>> :infer
              MemRef.copy(m1, m2) >>> []
              b = LLVM.load(term_ptr) >>> ENIF.Type.term()
              Func.return(b) >>> []
            end
          end
        end
      end
    end
  end
end
