defmodule Beaver.Nx.Defn do
  require Beaver
  import Beaver, only: [mlir: 1]
  require Beaver.MLIR.Dialect.{Func, SCF, Linalg}
  alias Beaver.MLIR
  alias MLIR.Type

  alias Beaver.MLIR.Dialect.{
    Builtin,
    Func,
    TOSA,
    Arith,
    SCF,
    Tensor,
    Bufferization,
    MemRef,
    Linalg
  }

  alias Beaver.MLIR.Dialect
  import Builtin, only: :macros
  import MLIR, only: :macros
  import MLIR.Sigils

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

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [pos]}})
       when is_integer(pos) do
    MLIR.Managed.Block.get() |> Beaver.MLIR.Block.get_arg!(pos)
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [:nan]},
           shape: {},
           type: {:f, 32}
         } = t
       ) do
    mlir do
      TOSA.const({:value, ~a{dense<0x7F800001> : tensor<f32>}}) :: ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [:infinity]},
           shape: {},
           type: {:f, 32}
         } = t
       ) do
    mlir do
      TOSA.const({:value, ~a{dense<0x7F800000> : tensor<f32>}}) ::
        ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [:neg_infinity]},
           shape: {},
           type: {:f, 32}
         } = t
       ) do
    mlir do
      TOSA.const({:value, ~a{dense<0xFF800000> : tensor<f32>}}) ::
        ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [value]},
           shape: {},
           type: type
         } = t
       )
       when is_integer(value) or is_float(value) do
    mlir do
      TOSA.const({:value, ~a{dense<#{value}> : tensor<#{get_type_name(type)}>}}) ::
        ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :constant, args: [%Complex{im: im, re: re}]},
           type: {:c, 64}
         } = t
       ) do
    mlir do
      Arith.constant({:value, ~a[dense<(#{re}, #{im})> : #{gen_type_str(t)}]}) ::
        ~t{#{gen_type_str(t)}}
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

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :negate, args: [input1]}} = t) do
    mlir do
      input1 = gen_op(input1)
      TOSA.negate(input1) :: ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(%Nx.Tensor{data: %Nx.Defn.Expr{op: :multiply, args: [a, b]}} = t) do
    mlir do
      a = gen_op(a)
      b = gen_op(b)
      TOSA.mul(a, b, shift: ~a{0}i32) :: ~t{#{gen_type_str(t)}}
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

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{
             op: :conjugate,
             args: [%Nx.Tensor{type: {:c, 64}} = complex_tensor]
           },
           shape: {}
         } = t
       ) do
    mlir do
      complex_tensor = gen_op(complex_tensor)
      complex_element = Tensor.extract(complex_tensor) :: ~t{complex<f32>}
      conjugate_element = Dialect.Complex.conj(complex_element) :: ~t{complex<f32>}

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ~a{dense<0> : vector<2xi32>}) ::
        ~t{#{gen_type_str(t)}}

      Tensor.insert(conjugate_element, conjugate_tensor) ::
        ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :conjugate, args: [%Nx.Tensor{} = real_tensor]},
           shape: {},
           type: complex_type = {:c, 64}
         } = t
       ) do
    mlir do
      real_tensor = gen_op(real_tensor)
      real_tensor = TOSA.cast(real_tensor) :: Type.ranked_tensor([], Type.f32())
      real = Tensor.extract(real_tensor) :: ~t{f32}

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ~a{dense<0> : vector<2xi32>}) ::
        ~t{#{gen_type_str(t)}}

      imaginary = Arith.constant(value: ~a{0.0}f32) :: ~t{f32}

      complex_element_t = ~t{#{get_type_name(complex_type)}}
      complex_element = Dialect.Complex.create(real, imaginary) :: complex_element_t
      conjugate_element = Dialect.Complex.conj(complex_element) :: complex_element_t

      Tensor.insert(conjugate_element, conjugate_tensor) :: ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{op: :conjugate, args: [complex_tensor]},
           shape: shape
         } = t
       ) do
    mlir do
      element_cnt = Enum.reduce(Tuple.to_list(shape), 1, &*/2)
      complex_tensor = gen_op(complex_tensor)
      index_t = ~t{index}
      lower = Arith.constant(value: ~a{0}index) :: index_t
      upper = Arith.constant(value: ~a{#{element_cnt}}index) :: index_t
      step = Arith.constant(value: ~a{1}index) :: index_t

      conjugate_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ~a{dense<0> : vector<2xi32>}) ::
        ~t{#{gen_type_str(t)}}

      conjugate_memref = Bufferization.to_memref(conjugate_tensor) ::
        Type.memref([2], Type.complex(Type.f32()))

      SCF.for [lower, upper, step] do
        region do
          block inner(index :: ~t{index}) do
            complex_element = Tensor.extract(complex_tensor, index) :: Type.complex(Type.f32())
            conjugate_element = Dialect.Complex.conj(complex_element) :: ~t{f32}complex
            MemRef.store([conjugate_element, conjugate_memref, index])
            SCF.yield(defer_if_terminator: false)
          end
        end
      end

      conjugate_tensor
    end
  end

  defp gen_op(
         %Nx.Tensor{
           data: %Nx.Defn.Expr{
             op: :imag,
             args: [%Nx.Tensor{type: {:c, 64}} = in_tensor]
           },
           shape: {}
         } = t
       ) do
    mlir do
      in_tensor = gen_op(in_tensor)
      out_t = ~t{#{gen_type_str(t)}}

      out_tensor = Bufferization.alloc_tensor(operand_segment_sizes: ~a{dense<0> : vector<2xi32>}) ::
        out_t

      Linalg.generic [
        in_tensor,
        out_tensor,
        operand_segment_sizes: ~a{dense<1> : vector<2xi32>},
        indexing_maps: ~a{[affine_map<() -> ()>, affine_map<() -> ()>]},
        iterator_types: ~a{[]}
      ] do
        region do
          block bb0(arg0 :: ~t<f32>complex, arg1 :: Type.f(32)) do
            im = Dialect.Complex.im(arg0) :: Type.f32()
            Linalg.yield([im, defer_if_terminator: false])
          end
        end
      end :: ~t{#{gen_type_str(t)}}
    end
  end

  defp gen_op(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&gen_op/1)
    |> List.to_tuple()
  end

  defp gen_op(tensor) do
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
          Func.func beaver_nx_main(
                      sym_name: "\"#{symbol}\"",
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

                  ret = %Beaver.MLIR.CAPI.IR.Value{} ->
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
      |> invoke(args, jit, symbol)

    [tree_return]
  end

  @doc """
  Invoke MLIR JIT with NX tensors. If there are tuples their memrefs will be packed into a single C struct.
  """

  def invoke(return, args, jit, symbol) do
    # pack the tensor tuples into a C struct
    jit_args = [return_struct | _] = [return | args] |> Enum.map(&memref_from_tensor/1)
    if List.improper?(jit_args), do: raise("jit arguments is not a proper list")

    MLIR.ExecutionEngine.invoke!(
      jit,
      symbol,
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

  @doc """
  Run passes to compile IR generated from NX expressions, mostly in TOSA and some LinAlg. The results should be in LLVM.
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
