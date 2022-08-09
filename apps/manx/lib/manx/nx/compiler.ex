defmodule Manx.Compiler do
  alias Beaver.MLIR
  import MLIR.Sigils
  import Beaver, only: :macros
  require Beaver.MLIR
  @behaviour Nx.Defn.Compiler
  @impl true
  def __jit__(key, vars, fun, [args], _options) do
    # call fun to generated tree
    tree = fun.(vars)

    info = Function.info(key)
    uniq = info |> Keyword.get(:uniq)
    module = info |> Keyword.get(:module)
    name = info |> Keyword.get(:name)

    symbol =
      Module.concat([module, name, "#{uniq}"])
      |> Atom.to_string()

    # generate ir
    entry_types =
      Enum.reduce(vars, [], fn
        tuple, acc when is_tuple(tuple) ->
          acc ++ Enum.map(Tuple.to_list(tuple), &Manx.Defn.gen_type/1)

        t, acc ->
          acc ++ [Manx.Defn.gen_type(t)]
      end)

    ir =
      mlir do
        module do
          function_type =
            Type.function(
              entry_types,
              Manx.Defn.gen_root_types(tree)
            )

          Func.func manx_main(
                      sym_name: "\"#{symbol}\"",
                      function_type: function_type
                    ) do
            region do
              entry =
                MLIR.Block.create(
                  entry_types,
                  List.duplicate(MLIR.Managed.Location.get(), length(entry_types))
                )

              root = Manx.Defn.gen_op(%Manx.Defn.Env{block: entry}, tree)

              mlir block: entry do
                case root do
                  ret = %Beaver.MLIR.Value{} ->
                    Func.return(ret) >>> []

                  tuple_ret when is_tuple(tuple_ret) ->
                    Func.return(Tuple.to_list(tuple_ret)) >>> []
                end
              end

              MLIR.__REGION__()
              |> Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(entry)
            end
          end
        end
      end

    # lower ir to llvm and create jit
    llvm_ir = ir |> Manx.Lowering.tosa_vulkan()
    # jit = MLIR.ExecutionEngine.create!(llvm_ir)

    jit =
      llvm_ir
      |> MLIR.ExecutionEngine.create!(
        shared_lib_paths: [
          Beaver.LLVM.Config.lib_dir() |> Path.join("libvulkan-runtime-wrappers.dylib"),
          Beaver.LLVM.Config.lib_dir() |> Path.join("libmlir_runner_utils.dylib")
        ]
      )

    # invoke jit and setting return for tree
    tree_return =
      tree
      |> Manx.tensor_of_null_memref()
      |> invoke(args, jit, symbol)

    [tree_return]
  end

  @doc """
  Invoke MLIR JIT with Nx tensors. If there are tuples their memrefs will be packed into a single C struct.
  """

  def invoke(return, args, jit, symbol) do
    # pack the tensor tuples into a C struct
    jit_args =
      [return_struct | _] =
      [return | args]
      |> Enum.map(&memref_from_tensor/1)

    if List.improper?(jit_args), do: raise("jit arguments is not a proper list")

    MLIR.ExecutionEngine.invoke!(
      jit,
      symbol,
      jit_args |> Enum.map(&Beaver.Native.Memory.descriptor_ptr/1)
    )

    # unpack the C struct into tensor tuples
    populate_tensor_from_memref(return, return_struct)
  end

  @doc """
  - If it is a tensor, return a memref
  - If it is a tuple, recursively pack them into one struct.
  """
  def memref_from_tensor(%Nx.Tensor{data: %Manx{memref: memref}}), do: memref

  def memref_from_tensor(
        %Nx.Tensor{
          data: %Nx.BinaryBackend{state: binary}
        } = tensor
      ) do
    Manx.from_binary(tensor, binary, []) |> memref_from_tensor
  end

  def memref_from_tensor({}) do
    raise "can't extract memref from an empty tuple"
  end

  def memref_from_tensor(tuple) when is_tuple(tuple) do
    mems =
      Tuple.to_list(tuple)
      |> Enum.map(&memref_from_tensor/1)

    # TODO: support array of memref descriptor of different kinds
    first = mems |> List.first()
    kind = first.descriptor.kind

    refs =
      mems
      |> Enum.map(fn %Beaver.Native.Memory{descriptor: %Beaver.Native.Memory.Descriptor{ref: ref}} ->
        ref
      end)

    # TODO: add a raw NIF beaver_raw_create_heterogeneous_array, using union maybe
    mut_array = Beaver.Native.forward(kind, :mut_array, [refs])

    struct!(Beaver.Native.Array,
      element_kind: kind,
      ref: mut_array
    )
  end

  @doc """
  - If it is a tensor, return a memref
  - If it is a tuple, recursively unpack each member from the nested struct.
  """
  def populate_tensor_from_memref(%Nx.Tensor{data: %Manx{}} = tensor, memref) do
    %{tensor | data: %Manx{memref: memref}}
  end

  def populate_tensor_from_memref(
        tuple,
        %Beaver.Native.Array{element_kind: element_kind} = nested_struct
      )
      when is_tuple(tuple) do
    nested_struct_ptr = nested_struct |> Beaver.Native.Memory.descriptor_ptr()

    {tensors, _offset} =
      Enum.reduce(tuple |> Tuple.to_list(), {[], 0}, fn x, {acc, offset} ->
        {ref, size} =
          Beaver.Native.OpaquePtr.to_resource(
            element_kind,
            nested_struct_ptr,
            offset
          )

        mem = %Beaver.Native.Memory{
          descriptor: %Beaver.Native.Memory.Descriptor{
            ref: ref,
            kind: element_kind
          }
        }

        {acc ++ [populate_tensor_from_memref(x, mem)], offset + size}
      end)

    tensors |> List.to_tuple()
  end
end
