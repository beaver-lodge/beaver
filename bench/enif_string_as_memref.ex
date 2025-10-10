defmodule ENIFStringAsMemRef do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias MLIR.Dialect.{Func, MemRef, Index, Arith, CF, Ptr}
  require Func
  use ENIFSupport

  defp dim0_size_as_i64(%Beaver.SSA{arguments: [m], ctx: ctx, blk: blk}) do
    mlir ctx: ctx, blk: blk do
      zero = Index.constant(value: Attribute.index(0)) >>> Type.index()
      len = MemRef.dim(m, zero) >>> :infer
      Index.casts(len) >>> ~t<i64>
    end
  end

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
              m = ENIF.inspect_binary_as_memref(env, s) >>> :infer
              len = dim0_size_as_i64(m) >>> :infer
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
              success = ENIF.is_binary(env, s) >>> :infer

              success =
                Arith.cmpi(success, one, predicate: Arith.cmp_i_predicate(:eq)) >>> Type.i1()

              CF.cond_br(success, Beaver.Env.block(successor), Beaver.Env.block(exit_blk)) >>> []
            end

            block exit_blk() do
              msg = MemRef.get_global(name: Attribute.flat_symbol_ref(global_s)) >>> global_t
              size = msg |> MLIR.Value.type() |> MLIR.ShapedType.dim_size(0)
              size = Arith.constant(value: Attribute.integer(Type.i64(), size)) >>> :infer

              term_ptr =
                MemRef.alloca(operand_segment_sizes: :infer) >>>
                  Type.memref!([], ENIF.Type.term(ctx: ctx), memory_space: ~a{#ptr.generic_space})

              term_ptr_arg = Ptr.to_ptr(term_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}
              # assign the term and return the pointer to binary data
              m = ENIF.make_new_binary_as_memref(env, size, term_ptr_arg) >>> :infer
              MemRef.copy(source: msg, target: m) >>> []
              # load the term from the memref and wrap it as exception
              msg = MemRef.load(term_ptr) >>> ENIF.Type.term()
              e = ENIF.raise_exception(env, msg) >>> []
              Func.return(e) >>> []
            end

            block successor() do
              m1 = ENIF.inspect_binary_as_memref(env, s) >>> :infer
              size = dim0_size_as_i64(m1) >>> :infer

              term_ptr =
                MemRef.alloca(operand_segment_sizes: :infer) >>>
                  Type.memref!([], ENIF.Type.term(ctx: ctx), memory_space: ~a{#ptr.generic_space})

              term_ptr_arg = Ptr.to_ptr(term_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}
              m2 = ENIF.make_new_binary_as_memref(env, size, term_ptr_arg) >>> :infer
              MemRef.copy(source: m1, target: m2) >>> []
              b = MemRef.load(term_ptr) >>> ENIF.Type.term()
              Func.return(b) >>> []
            end
          end
        end
      end
    end
  end
end
