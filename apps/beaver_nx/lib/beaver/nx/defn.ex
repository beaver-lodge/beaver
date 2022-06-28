defmodule Beaver.Nx.Defn do
  require Beaver
  import Beaver, only: [mlir: 1]
  require Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.{Builtin, Func, TOSA}
  import Builtin, only: :macros
  import MLIR, only: :macros
  import MLIR.Sigils

  defp get_type_name(:s), do: "i"

  defp get_type_name(:f), do: "f"

  # TODO: stop using string interpolation because it is essentially a hack
  defp gen_type_str(%Nx.Tensor{shape: {}, type: {name, size}}) do
    "tensor<#{get_type_name(name)}#{size}>"
  end

  defp gen_type_str(%Nx.Tensor{shape: {dim0}, type: {name, size}}) do
    "tensor<#{dim0}x#{get_type_name(name)}#{size}>"
  end

  defp gen_type_str(%Nx.Tensor{shape: {dim0, dim1}, type: {name, size}}) do
    "tensor<#{dim0}x#{dim1}x#{get_type_name(name)}#{size}>"
  end

  defp gen_type_str(tuple) when is_tuple(tuple) do
    joined =
      Tuple.to_list(tuple)
      |> Enum.map(&gen_type_str/1)
      |> Enum.join(", ")

    "(" <> joined <> ")"
  end

  defp gen_type_str(t) do
    raise "type unsupported: " <> inspect(t, structs: false, pretty: true)
  end

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}})
       when is_integer(pos) do
    MLIR.Managed.Block.get() |> Beaver.MLIR.Block.get_arg!(pos)
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [int_value]},
           shape: {},
           type: {:s, size}
         } = t
       )
       when is_integer(int_value) do
    mlir do
      TOSA.const({:value, ~a{dense<#{int_value}> : tensor<i#{size}>}}) :: ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{
             args: [%Nx.Tensor{data: %Nx.BinaryBackend{state: binary}}],
             op: :tensor
           }
         } = t
       ) do
    mlir do
      raw_buffer = Exotic.Value.Struct.get(binary)

      tensor_attr =
        MLIR.CAPI.mlirDenseElementsAttrRawBufferGet(
          ~t{#{gen_type_str(t)}},
          Exotic.Value.get(:isize, byte_size(binary)),
          Exotic.Value.get_ptr(raw_buffer)
        )

      if MLIR.Attribute.is_null(tensor_attr), do: raise("fail to parse tensor dense elements")

      TOSA.const({:value, tensor_attr}) :: ~t{#{gen_type_str(t)}}
    end
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

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :add, args: [a, b]}} = t) do
    mlir do
      a = gen_op(a)
      b = gen_op(b)
      TOSA.add(a, b) :: ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :subtract, args: [a, b]}} = t) do
    mlir do
      a = gen_op(a)
      b = gen_op(b)
      TOSA.sub(a, b) :: ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&gen_op/1)
    |> List.to_tuple()
  end

  defp gen_op(tensor) do
    raise "op unsupported: " <> inspect(tensor, structs: false, pretty: true)
  end

  @doc false
  def __jit__(_key, vars, fun, [args], _options) do
    # call fun to generated tree
    tree = fun.(vars)

    # generate ir
    ir =
      mlir do
        module do
          Func.func beaver_nx_main(
                      function_type:
                        ~a"#{gen_type_str(List.to_tuple(vars))} -> #{gen_type_str(tree)}"
                    ) do
            region do
              block =
                for arg <- vars do
                  {~t{#{gen_type_str(arg)}}, MLIR.Managed.Location.get()}
                end
                |> MLIR.Block.create()

              MLIR.Block.under(block, fn ->
                case gen_op(tree) do
                  ret = %Beaver.MLIR.CAPI.MlirValue{} ->
                    Func.return(ret)

                  tuple_ret when is_tuple(tuple_ret) ->
                    Func.return(Tuple.to_list(tuple_ret))
                end
              end)

              MLIR.Managed.Region.get()
              |> Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(block)
            end
          end
        end
      end

    # lower ir to llvm and create jit
    llvm_ir = ir |> tosa_cpu()
    jit = MLIR.ExecutionEngine.create!(llvm_ir)

    # invoke jit and setting return for tree
    tree_return =
      tree
      |> Beaver.Nx.tensor_of_null_memref()
      |> invoke(args, jit)

    [tree_return]
  end

  @doc """
  Invoke MLIR JIT with NX tensors. If there are tuples their memrefs will be packed into a single C struct.
  """
  def invoke(return, args, jit) do
    # pack the tensor tuples into a C struct
    jit_args = [return_struct | _] = [return | args] |> Enum.map(&memref_from_tensor/1)
    if List.improper?(jit_args), do: raise("jit arguments is not a proper list")

    MLIR.ExecutionEngine.invoke!(
      jit,
      "beaver_nx_main",
      Enum.map(jit_args, &Exotic.Value.get_ptr/1)
    )

    # unpack the C struct into tensor tuples
    populate_tensor_from_memref(return, return_struct)
  end

  @doc """
  - If it is a tensor, return a memref
  - If it is a tuple, recursively pack them into one struct.
  """
  def memref_from_tensor(%Nx.Tensor{data: %Beaver.Nx{memref: memref}}), do: memref

  def memref_from_tensor(
        %Nx.Tensor{
          data: %Nx.BinaryBackend{state: binary}
        } = tensor
      ) do
    Beaver.Nx.from_binary(tensor, binary, []) |> memref_from_tensor
  end

  def memref_from_tensor({}) do
    raise "can't extract memref from an empty tuple"
  end

  def memref_from_tensor(tuple) when is_tuple(tuple) do
    # convert to a list of memrefs and fields
    {list_of_fields, memrefs} =
      for tensor <- Tuple.to_list(tuple) do
        tensor |> memref_from_tensor
      end
      # TODO: code here is really ugly
      |> Enum.reduce({[], []}, fn %Exotic.Value.Struct{fields: fields} = memref,
                                  {fields_acc, memref_acc} ->
        {
          fields_acc ++ [fields],
          memref_acc ++ [memref]
        }
      end)

    # generate fields for the nested struct
    list_of_fields =
      for {fields, i} <- Enum.with_index(list_of_fields) do
        {String.to_atom("packed_struct_#{i}"), {:struct, fields}}
      end

    Exotic.Value.Struct.get(list_of_fields, memrefs)
  end

  @doc """
  - If it is a tensor, return a memref
  - If it is a tuple, recursively unpack each member from the nested struct.
  """
  def populate_tensor_from_memref(%Nx.Tensor{data: %Beaver.Nx{}} = tensor, memref) do
    %{tensor | data: %Beaver.Nx{memref: memref}}
  end

  def populate_tensor_from_memref(tuple, %Exotic.Value.Struct{fields: fields} = nested_struct)
      when is_tuple(tuple) do
    Enum.zip(Tuple.to_list(tuple), fields)
    |> Enum.map(fn {tensor, {packed_field_name, _}} ->
      populate_tensor_from_memref(
        tensor,
        Exotic.Value.fetch(nested_struct, packed_field_name)
      )
    end)
    |> List.to_tuple()
  end

  def tosa_cpu(op) do
    import MLIR.{Transforms, Conversion}

    op
    |> MLIR.Operation.verify!()
    |> canonicalize
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "tosa-layerwise-constant-fold")
    end)
    |> cse
    |> MLIR.Pass.Composer.run!(dump: false)
    |> tosa_to_scf
    |> tosa_to_arith
    |> tosa_to_tensor()
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "tosa-make-broadcastable")
    end)
    |> convert_tensor_to_linalg()
    |> MLIR.Pass.Composer.nested("func.func", [
      tosa_to_linalg(),
      linalg_fuse_elementwise_ops()
    ])
    |> MLIR.Pass.Composer.nested("func.func", [
      linalg_bufferize(),
      convert_linalg_to_loops(),
      lower_affine(),
      convert_math_to_llvm(),
      convert_arith_to_llvm(),
      convert_scf_to_cf(),
      "arith-expand",
      "memref-expand"
    ])
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "tensor-bufferize")
    end)
    |> MLIR.Pass.Composer.pipeline("arith-bufferize,func-bufferize")
    |> MLIR.Pass.Composer.nested("func.func", fn pm ->
      MLIR.Pass.pipeline!(pm, "llvm-request-c-wrappers")
    end)
    |> convert_vector_to_llvm
    |> convert_memref_to_llvm
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run!(dump_if_fail: true)
  end
end
