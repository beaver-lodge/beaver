defmodule ENIFStringAsMemRef do
  @moduledoc false
  alias Beaver.ENIF
  use Beaver
  alias MLIR.Dialect.{Func, MemRef, Index, LLVM, Arith, CF, Builtin, Ptr}
  require Func
  use ENIFSupport

  defp ptr_as_memref(%Beaver.SSA{arguments: [ptr, size], ctx: ctx, blk: blk}) do
    mlir ctx: ctx, blk: blk do
      size = Index.casts(size) >>> Type.index()

      m =
        Ptr.from_ptr(ptr) >>>
          Type.unranked_memref(Type.i8(), memory_space: ~a{#ptr.generic_space})

      MemRef.reinterpret_cast(
        source: m,
        sizes: size,
        operand_segment_sizes: :infer,
        static_offsets: Attribute.dense_array([0], Beaver.Native.I64, ctx: ctx),
        static_sizes:
          Attribute.dense_array([MLIR.ShapedType.dynamic_size()], Beaver.Native.I64, ctx: ctx),
        static_strides: Attribute.dense_array([1], Beaver.Native.I64, ctx: ctx)
      ) >>>
        Type.memref!([:dynamic], Type.i8(ctx: ctx),
          layout: MLIR.Attribute.strided_layout(0, [1], ctx: ctx),
          memory_space: ~a{#ptr.generic_space}
        )
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
              one = Arith.constant(value: Attribute.integer(Type.i(32), 1)) >>> :infer
              b_ptr = LLVM.alloca(one, elem_type: ENIF.Type.binary()) >>> ~t{!llvm.ptr}

              b_ptr_arg =
                Builtin.unrealized_conversion_cast(b_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}

              ENIF.inspect_binary(env, s, b_ptr_arg) >>> :infer
              b = LLVM.load(b_ptr) >>> ENIF.Type.binary()
              size = LLVM.extractvalue(b, position: ~a{array<i64: 0>}) >>> Type.i64()
              d_ptr = LLVM.extractvalue(b, position: ~a{array<i64: 1>}) >>> ~t{!llvm.ptr}

              d_ptr_arg =
                Builtin.unrealized_conversion_cast(d_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}

              m = ptr_as_memref(d_ptr_arg, size) >>> :infer
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

              b_ptr_arg =
                Builtin.unrealized_conversion_cast(b_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}

              success = ENIF.inspect_binary(env, s, b_ptr_arg) >>> :infer

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
              d_ptr = ENIF.make_new_binary(env, size, term_ptr_arg) >>> :infer
              # copy the error message to the binary data
              m = ENIF.ptr_to_memref(d_ptr, size) >>> :infer
              MemRef.copy(source: msg, target: m) >>> []
              # load the term from the memref and wrap it as exception
              msg = MemRef.load(term_ptr) >>> ENIF.Type.term()
              e = ENIF.raise_exception(env, msg) >>> []
              Func.return(e) >>> []
            end

            block successor() do
              b = LLVM.load(b_ptr) >>> ENIF.Type.binary()
              size = LLVM.extractvalue(b, position: ~a{array<i64: 0>}) >>> Type.i64()
              d_ptr = LLVM.extractvalue(b, position: ~a{array<i64: 1>}) >>> ~t{!llvm.ptr}

              d_ptr_arg =
                Builtin.unrealized_conversion_cast(d_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}

              m1 = ENIF.ptr_to_memref(d_ptr_arg, size) >>> :infer

              term_ptr =
                MemRef.alloca(operand_segment_sizes: :infer) >>>
                  Type.memref!([], ENIF.Type.term(ctx: ctx), memory_space: ~a{#ptr.generic_space})

              term_ptr_arg = Ptr.to_ptr(term_ptr) >>> ~t{!ptr.ptr<#ptr.generic_space>}
              d_ptr = ENIF.make_new_binary(env, size, term_ptr_arg) >>> :infer
              m2 = ENIF.ptr_to_memref(d_ptr, size) >>> :infer
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
