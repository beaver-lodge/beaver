defmodule Beaver.Nx.Defn do
  require Beaver
  import Beaver, only: [mlir: 1]
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.{Builtin, Func, TOSA}
  import Builtin, only: :macros
  import MLIR, only: :macros
  import MLIR.Sigils

  defp gen_type_str(%Nx.Tensor{shape: {}, type: {:s, size}}) do
    "tensor<i#{size}>"
  end

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}})
       when is_integer(pos) do
    MLIR.Managed.Block.get() |> Beaver.MLIR.Block.get_arg!(pos)
  end

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :multiply, args: [a, b]}} = t) do
    mlir do
      a = gen_op(a)
      b = gen_op(b)
      TOSA.mul(a, b, {:shift, ~a{0 : i32}}) :: ~t{#{gen_type_str(t)}}
    end
  end

  @doc false
  def __jit__(_key, vars, fun, [args], _options) do
    tree = fun.(vars)

    arg_types_str =
      for arg <- vars do
        gen_type_str(arg)
      end
      |> Enum.join(", ")

    entry_block_args =
      for arg <- vars do
        {~t{#{gen_type_str(arg)}}, MLIR.Managed.Location.get()}
      end

    return_t = gen_type_str(tree)

    ir =
      mlir do
        module do
          Func.func beaver_nx_main(function_type: ~a"(#{arg_types_str}) -> (#{return_t})") do
            region do
              block = MLIR.Block.create(entry_block_args)

              MLIR.Block.under(block, fn ->
                ret = %Beaver.MLIR.CAPI.MlirValue{} = gen_op(tree)
                Func.return(ret)
              end)

              region = MLIR.Managed.Region.get()
              Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(region, block)
            end
          end
        end
      end

    llvm_ir = ir |> tosa_cpu()
    jit = MLIR.ExecutionEngine.create!(llvm_ir)

    arg_memrefs =
      for a <- args do
        a |> memref_from_tensor
      end

    # create a tensor containing memref and setting all the bits to 0
    # TODO: create a memref with a null ptr
    return_tensor = tree |> Beaver.Nx.tensor_of_null_memref()
    return_memref = return_tensor |> memref_from_tensor

    jit_args = [return_memref | arg_memrefs]
    if List.improper?(jit_args), do: raise("jit arguments is not a proper list")
    jit_args = Enum.map(jit_args, &Exotic.Value.get_ptr/1)

    MLIR.ExecutionEngine.invoke!(jit, "beaver_nx_main", jit_args)

    [return_tensor]
  end

  import MLIR.{Transforms, Conversion}

  def memref_from_tensor(%Nx.Tensor{data: %Beaver.Nx{memref: memref}}), do: memref

  def tosa_cpu(op) do
    op
    |> MLIR.Operation.verify!()
    |> canonicalize
    |> cse
    |> tosa_to_scf
    |> tosa_to_arith
    |> tosa_to_tensor()
    |> convert_tensor_to_linalg()
    |> MLIR.Pass.Composer.nested("func.func", [
      tosa_to_linalg(),
      linalg_fuse_elementwise_ops(),
      linalg_bufferize(),
      convert_linalg_to_loops(),
      lower_affine(),
      convert_math_to_llvm(),
      convert_scf_to_cf(),
      "arith-expand",
      "memref-expand"
    ])
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "tensor-bufferize")
    end)
    |> MLIR.Pass.Composer.pipeline("func-bufferize")
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "llvm-request-c-wrappers")
    end)
    |> convert_vector_to_llvm
    |> convert_memref_to_llvm
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run!()
  end
end
