defmodule Beaver.Nx.Defn do
  defmodule Env do
    defstruct block: nil
  end

  use Beaver
  alias Beaver.MLIR
  alias MLIR.{Type, Attribute}

  defp gen_type({:s, size}), do: Type.i(size)
  defp gen_type({:f, size}), do: Type.f(size)
  defp gen_type({:c, size}), do: Type.complex(Type.f(div(size, 2)))

  defp gen_type(%Nx.Tensor{shape: shape, type: type}) do
    Tuple.to_list(shape)
    |> Type.ranked_tensor(gen_type(type))
  end

  defp gen_type(tuple) when is_tuple(tuple) do
    Tuple.to_list(tuple)
    |> Enum.map(&gen_type/1)
    |> Type.tuple()
  end

  # In upstream MLIR, there is no lower-able Op packing multiple values into a tuple.
  # If the Nx root type is a tuple, it should be converted to repeated results.
  # This function should always return a list of types
  defp gen_root_types(tuple) when is_tuple(tuple) do
    Tuple.to_list(tuple)
    |> Enum.map(&gen_type/1)
  end

  defp gen_root_types(type), do: [gen_type(type)]

  defp get_type_name({:s, size}), do: "i#{size}"

  defp get_type_name({:f, size}), do: "f#{size}"

  defp get_type_name({:c, size}) do
    "complex<f#{div(size, 2)}>"
  end

  # TODO: stop using string interpolation because it is essentially a hack
  defp gen_type_str(%Nx.Tensor{shape: {}, type: type}) do
    "tensor<#{get_type_name(type)}>"
  end

  defp gen_type_str(%Nx.Tensor{shape: {dim0}, type: type}) do
    "tensor<#{dim0}x#{get_type_name(type)}>"
  end

  defp gen_type_str(%Nx.Tensor{shape: {dim0, dim1}, type: type}) do
    "tensor<#{dim0}x#{dim1}x#{get_type_name(type)}>"
  end

  defp gen_type_str(%Nx.Tensor{shape: {dim0, dim1, dim2}, type: type}) do
    "tensor<#{dim0}x#{dim1}x#{dim2}x#{get_type_name(type)}>"
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

  defp gen_op(%Env{block: block}, %Nx.Tensor{
         data: %Nx.Defn.Expr{op: :parameter, args: [pos]}
       })
       when is_integer(pos) do
    block |> Beaver.MLIR.Block.get_arg!(pos)
  end

  defp gen_op(
         %Env{block: block},
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [:nan]},
           shape: {},
           type: {:f, 32}
         } = t
       ) do
    mlir block: block do
      TOSA.const({:value, ~a{dense<0x7F800001> : tensor<f32>}}) >>> gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block},
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [:infinity]},
           shape: {},
           type: {:f, 32}
         } = t
       ) do
    mlir block: block do
      TOSA.const({:value, ~a{dense<0x7F800000> : tensor<f32>}}) >>>
        ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Env{block: block},
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [:neg_infinity]},
           shape: {},
           type: {:f, 32}
         } = t
       ) do
    mlir block: block do
      _r =
        TOSA.const({:value, ~a{dense<0xFF800000> : tensor<f32>}}) >>>
          gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block},
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [value]},
           shape: {},
           type: type
         } = t
       )
       when is_integer(value) or is_float(value) do
    mlir block: block do
      _r =
        TOSA.const({:value, ~a{dense<#{value}> : tensor<#{get_type_name(type)}>}}) >>>
          gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block},
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [%Complex{im: im, re: re}]},
           type: {:c, 64}
         } = t
       ) do
    mlir block: block do
      Arith.constant({:value, ~a[dense<(#{re}, #{im})> : #{gen_type_str(t)}]}) >>>
        ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Env{block: block},
         %Nx.Tensor{
           data: %Nx.Defn.Expr{
             args: [%Nx.Tensor{data: %Nx.BinaryBackend{state: binary}}],
             op: :tensor
           }
         } = t
       ) do
    mlir block: block do
      tensor_attr =
        MLIR.CAPI.mlirDenseElementsAttrRawBufferGet(
          gen_type(t),
          byte_size(binary),
          Beaver.Native.c_string(binary) |> Beaver.Native.Array.as_opaque()
        )

      if MLIR.Attribute.is_null(tensor_attr), do: raise("fail to parse tensor dense elements")

      TOSA.const({:value, tensor_attr}) >>> gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{data: %Nx.Defn.Expr{op: :negate, args: [input1]}} = t
       ) do
    mlir block: block do
      input1 = gen_op(env, input1)
      TOSA.negate(input1) >>> gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{data: %Nx.Defn.Expr{op: :multiply, args: [a, b]}} = t
       ) do
    mlir block: block do
      a = gen_op(env, a)
      b = gen_op(env, b)
      TOSA.mul(a, b, shift: Attribute.integer(Type.i(32), 0)) >>> gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{data: %Nx.Defn.Expr{op: :add, args: [a, b]}} = t
       ) do
    mlir block: block do
      a = gen_op(env, a)
      b = gen_op(env, b)
      TOSA.add(a, b) >>> gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{data: %Nx.Defn.Expr{op: :subtract, args: [a, b]}} = t
       ) do
    mlir block: block do
      a = gen_op(env, a)
      b = gen_op(env, b)
      TOSA.sub(a, b) >>> gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{
           data: %Nx.Defn.Expr{
             op: :conjugate,
             args: [%Nx.Tensor{type: {:c, 64}} = complex_tensor]
           },
           shape: {}
         } = t
       ) do
    mlir block: block do
      complex_tensor = gen_op(env, complex_tensor)
      complex_element = Tensor.extract(complex_tensor) >>> Type.complex(Type.f32())
      conjugate_element = Complex.conj(complex_element) >>> Type.complex(Type.f32())

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0])) >>>
          gen_type(t)

      Tensor.insert(conjugate_element, conjugate_tensor) >>>
        gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :conjugate, args: [%Nx.Tensor{} = real_tensor]},
           shape: {},
           type: complex_type = {:c, 64}
         } = t
       ) do
    mlir block: block do
      real_tensor = gen_op(env, real_tensor)
      real_tensor = TOSA.cast(real_tensor) >>> Type.ranked_tensor([], Type.f32())
      real = Tensor.extract(real_tensor) >>> Type.f32()

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0])) >>>
          gen_type(t)

      imaginary = Arith.constant(value: Attribute.float(Type.f32(), 0.0)) >>> Type.f32()

      complex_element_t = gen_type(complex_type)
      complex_element = Complex.create(real, imaginary) >>> complex_element_t
      conjugate_element = Complex.conj(complex_element) >>> complex_element_t

      _ = Tensor.insert(conjugate_element, conjugate_tensor) >>> gen_type(t)
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :conjugate, args: [complex_tensor]},
           shape: shape
         } = t
       ) do
    mlir block: block do
      element_cnt = Enum.reduce(Tuple.to_list(shape), 1, &*/2)
      complex_tensor = gen_op(env, complex_tensor)
      lower = Arith.constant(value: Attribute.integer(Type.index(), 0)) >>> Type.index()
      upper = Arith.constant(value: Attribute.integer(Type.index(), element_cnt)) >>> Type.index()
      step = Arith.constant(value: Attribute.integer(Type.index(), 1)) >>> Type.index()

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0])) >>>
          gen_type(t)

      conjugate_memref =
        Bufferization.to_memref(conjugate_tensor) >>>
          Type.memref([2], Type.complex(Type.f32()))

      SCF.for [lower, upper, step] do
        region do
          block inner(index >>> Type.index()) do
            complex_element = Tensor.extract(complex_tensor, index) >>> Type.complex(Type.f32())
            conjugate_element = Complex.conj(complex_element) >>> Type.complex(Type.f32())
            MemRef.store([conjugate_element, conjugate_memref, index]) >>> []
            SCF.yield() >>> []
          end
        end
      end >>> []

      conjugate_tensor
    end
  end

  defp gen_op(
         %Env{block: block} = env,
         %Nx.Tensor{
           data: %Nx.Defn.Expr{
             op: :imag,
             args: [%Nx.Tensor{type: {:c, 64}} = in_tensor]
           },
           shape: {}
         } = t
       ) do
    mlir block: block do
      in_tensor = gen_op(env, in_tensor)

      out_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0])) >>>
          gen_type(t)

      Linalg.generic [
        in_tensor,
        out_tensor,
        operand_segment_sizes: ODS.operand_segment_sizes([1, 1]),
        indexing_maps: ~a{[affine_map<() -> ()>, affine_map<() -> ()>]},
        iterator_types: ~a{[]}
      ] do
        region do
          block bb0(arg0 >>> Type.complex(Type.f32()), arg1 >>> Type.f(32)) do
            %MLIR.Value{} = arg1
            im = Complex.im(arg0) >>> Type.f32()
            Linalg.yield([im]) >>> []
          end
        end
      end >>> gen_type(t)
    end
  end

  defp gen_op(%Env{} = env, tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&gen_op(env, &1))
    |> List.to_tuple()
  end

  defp gen_op(_, tensor) do
    raise "op not supported: " <> inspect(tensor, structs: false, pretty: true)
  end

  @doc false
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
    ir =
      mlir do
        module do
          function_type =
            Type.function(
              Enum.map(vars, &gen_type/1),
              gen_root_types(tree)
            )

          Func.func beaver_nx_main(
                      sym_name: "\"#{symbol}\"",
                      function_type: function_type
                    ) do
            region do
              entry =
                for arg <- vars do
                  {gen_type(arg), MLIR.Managed.Location.get()}
                end
                |> MLIR.Block.create()

              root = gen_op(%Env{block: entry}, tree)

              mlir block: entry do
                case root do
                  ret = %Beaver.MLIR.Value{} ->
                    Func.return(ret) >>> []

                  tuple_ret when is_tuple(tuple_ret) ->
                    Func.return(Tuple.to_list(tuple_ret)) >>> []
                end
              end

              Beaver.Env.mlir__REGION__()
              |> Beaver.MLIR.CAPI.mlirRegionAppendOwnedBlock(entry)
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
    mems =
      Tuple.to_list(tuple)
      |> Enum.map(&memref_from_tensor/1)
      |> Enum.map(&Beaver.Native.Memory.descriptor_dump/1)

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
  def populate_tensor_from_memref(%Nx.Tensor{data: %Beaver.Nx{}} = tensor, memref) do
    %{tensor | data: %Beaver.Nx{memref: memref}}
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

  @doc """
  Run passes to compile IR generated from Nx expressions, mostly in TOSA and some LinAlg. The results should be in LLVM.
  """
  def tosa_cpu(op) do
    import MLIR.{Transforms, Conversion}

    op
    |> MLIR.Operation.verify!(dump_if_fail: true)
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
    |> MLIR.Pass.Composer.run!(dump: false)
    |> convert_complex_to_standard()
    |> convert_vector_to_llvm
    |> convert_memref_to_llvm
    |> convert_complex_to_llvm()
    |> convert_func_to_llvm
    |> reconcile_unrealized_casts
    |> MLIR.Pass.Composer.run!(dump_if_fail: true)
  end
end
