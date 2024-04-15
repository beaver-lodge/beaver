defmodule EnifTest do
  require Beaver.Env
  alias Beaver.ENIF
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias MLIR.{Attribute, Type}
  alias MLIR.Dialect.{Func, Arith, LLVM}
  import MLIR.Conversion
  require Func
  @moduletag :smoke
  test "populate enif functions", test_context do
    mlir ctx: test_context[:ctx] do
      module do
        Beaver.ENIF.populate_external_functions(Beaver.Env.context(), Beaver.Env.block())
        env_t = ENIF.Type.env()
        term_t = ENIF.Type.term()

        Func.func add(function_type: Type.function([env_t, term_t, term_t], [term_t])) do
          region do
            block _(env >>> env_t, left >>> term_t, right >>> term_t) do
              size = Attribute.integer(Type.i32(), 64)
              size = Arith.constant(value: size) >>> ~t<i32>
              left_ptr = LLVM.alloca(size, elem_type: Type.i64()) >>> ~t{!llvm.ptr}
              right_ptr = LLVM.alloca(size, elem_type: Type.i64()) >>> ~t{!llvm.ptr}
              {_, result_t} = Beaver.ENIF.signature(Beaver.Env.context(), :enif_get_int64)
              symbol = Attribute.flat_symbol_ref(:enif_get_int64)
              Func.call(env, left, left_ptr, callee: symbol) >>> result_t
              Func.call(env, right, right_ptr, callee: symbol) >>> result_t
              left = LLVM.load(left_ptr) >>> Type.i64()
              right = LLVM.load(right_ptr) >>> Type.i64()
              sum = Arith.addi(left, right) >>> Type.i64()
              {_, sum_t} = Beaver.ENIF.signature(Beaver.Env.context(), :enif_make_int64)
              symbol = Attribute.flat_symbol_ref(:enif_make_int64)
              sum = Func.call(env, sum, callee: symbol) >>> sum_t
              Func.return(sum) >>> []
            end
          end
        end
      end
    end
    |> MLIR.Operation.verify!()
    |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> convert_scf_to_cf
    |> convert_arith_to_llvm()
    |> convert_index_to_llvm()
    |> convert_func_to_llvm()
    |> MLIR.Pass.Composer.append("finalize-memref-to-llvm")
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run!()
    |> tap(fn m ->
      m
      |> MLIR.ExecutionEngine.create!(opt_level: 3)
      |> tap(fn %MLIR.ExecutionEngine{ref: jit_ref} ->
        MLIR.CAPI.beaver_raw_jit_register_enif(jit_ref)
        f = &MLIR.CAPI.beaver_raw_jit_invoke_with_terms(jit_ref, "add", [&1, &2])
        assert 3 == f.(1, 2)
        assert 1 == f.(-1, 2)
      end)
      |> MLIR.ExecutionEngine.destroy()
    end)
    |> MLIR.Module.destroy()
  end
end
