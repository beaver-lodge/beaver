defmodule Manx.Defn do
  defmodule Env do
    defstruct block: nil
  end

  alias Beaver.MLIR
  import MLIR.Sigils
  import Beaver, only: :macros
  require Beaver.MLIR
  alias MLIR.{Type, Attribute}

  def gen_type({:u, size}), do: Type.i(size)
  def gen_type({:s, size}), do: Type.i(size)
  def gen_type({:f, size}), do: Type.f(size)
  def gen_type({:c, size}), do: Type.complex(Type.f(div(size, 2)))

  def gen_type(%Nx.Tensor{shape: shape, type: type}) do
    Tuple.to_list(shape)
    |> Type.ranked_tensor(gen_type(type))
  end

  def gen_type(tuple) when is_tuple(tuple) do
    Tuple.to_list(tuple)
    |> Enum.map(&gen_type/1)
    |> Type.tuple()
  end

  # In upstream MLIR, there is no lower-able Op packing multiple values into a tuple.
  # If the Nx root type is a tuple, it should be converted to repeated results.
  # This function should always return a list of types
  def gen_root_types(tuple) when is_tuple(tuple) do
    Tuple.to_list(tuple)
    |> Enum.map(&gen_type/1)
  end

  def gen_root_types(type), do: [gen_type(type)]

  defp get_type_name({:s, size}), do: "i#{size}"

  defp get_type_name({:f, size}), do: "f#{size}"

  defp get_type_name({:c, size}) do
    "complex<f#{div(size, 2)}>"
  end

  # TODO: stop using string interpolation because it is essentially a hack
  def gen_type_str(%Nx.Tensor{shape: {}, type: type}) do
    "tensor<#{get_type_name(type)}>"
  end

  def gen_type_str(%Nx.Tensor{shape: {dim0}, type: type}) do
    "tensor<#{dim0}x#{get_type_name(type)}>"
  end

  def gen_type_str(%Nx.Tensor{shape: {dim0, dim1}, type: type}) do
    "tensor<#{dim0}x#{dim1}x#{get_type_name(type)}>"
  end

  def gen_type_str(%Nx.Tensor{shape: {dim0, dim1, dim2}, type: type}) do
    "tensor<#{dim0}x#{dim1}x#{dim2}x#{get_type_name(type)}>"
  end

  def gen_type_str(tuple) when is_tuple(tuple) do
    joined =
      Tuple.to_list(tuple)
      |> Enum.map(&gen_type_str/1)
      |> Enum.join(", ")

    "(" <> joined <> ")"
  end

  def gen_type_str(t) do
    raise "type unsupported: " <> inspect(t, structs: false, pretty: true)
  end

  defp gen_indexing_maps({}, {}) do
    ~a{[affine_map<() -> ()>, affine_map<() -> ()>]}
  end

  defp gen_indexing_maps({dim}, {dim}) do
    ~a{[affine_map<(d) -> (d)>, affine_map<(d) -> (d)>]}
  end

  defp gen_iterator_types({}, {}) do
    ~a{[]}
  end

  defp gen_iterator_types({dim}, {dim}) do
    ~a{["parallel"]}
  end

  defp gen_cast(%Env{block: block}, value, result_type) do
    value_type = MLIR.CAPI.mlirValueGetType(value)
    is_same_type = MLIR.CAPI.mlirTypeEqual(value_type, result_type) |> Beaver.Native.to_term()
    if is_same_type do
      value
    else
      mlir block: block do
        TOSA.cast(value) >>> result_type
      end
    end
  end

  def gen_op(%Env{block: block}, %Nx.Tensor{
        data: %Nx.Defn.Expr{op: :parameter, args: [pos]}
      })
      when is_integer(pos) do
    block |> Beaver.MLIR.Block.get_arg!(pos)
  end

  def gen_op(
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

  def gen_op(
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

  def gen_op(
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

  def gen_op(
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

  def gen_op(
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

  def gen_op(
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

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :negate, args: [input1]}} = t
      ) do
    mlir block: block do
      input1 = gen_op(env, input1)
      TOSA.negate(input1) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :multiply, args: [a, b]}} = t
      ) do
    mlir block: block do
      a = gen_op(env, a)
      b = gen_op(env, b)
      TOSA.mul(a, b, shift: Attribute.integer(Type.i(32), 0)) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :add, args: [a, b]}} = t
      ) do
    mlir block: block do
      a_t = gen_type(%{t| shape: a.shape})
      b_t = gen_type(%{t| shape: b.shape})
      a = gen_op(env, a)
      b = gen_op(env, b)
      a = gen_cast(env, a, a_t)
      b = gen_cast(env, b, b_t)
      TOSA.add(a, b) >>> gen_type(t)
    end
  end

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :subtract, args: [a, b]}} = t
      ) do
    mlir block: block do
      a = gen_op(env, a)
      b = gen_op(env, b)
      TOSA.sub(a, b) >>> gen_type(t)
    end
  end

  def gen_op(
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

  def gen_op(
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

  def gen_op(
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

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :imag,
            args: [%Nx.Tensor{type: {:c, 64}, shape: in_shape} = in_tensor]
          },
          shape: out_shape
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
        indexing_maps: gen_indexing_maps(in_shape, out_shape),
        iterator_types: gen_iterator_types(in_shape, out_shape)
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

  def gen_op(%Env{} = env, tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&gen_op(env, &1))
    |> List.to_tuple()
  end

  def gen_op(_, tensor) do
    raise "op not supported: " <> inspect(tensor, structs: false, pretty: true)
  end
end
