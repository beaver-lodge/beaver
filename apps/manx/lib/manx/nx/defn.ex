defmodule Manx.Defn do
  alias __MODULE__.Env
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

  @doc """
  In upstream MLIR, there is no lower-able Op packing multiple values into a tuple.
  If the Nx root type is a tuple, it should be converted to multi-results.
  This function should always return a list of types
  """
  def gen_root_types(tuple) when is_tuple(tuple) do
    Tuple.to_list(tuple)
    |> Enum.map(&gen_type/1)
  end

  def gen_root_types(type), do: [gen_type(type)]

  defp gen_affine_map(shape) do
    import MLIR.AffineMap
    rank = tuple_size(shape)

    exprs =
      shape
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.map(fn
        {1, _index} -> 0
        {dim_size, index} when dim_size > 1 -> dim(index)
      end)

    MLIR.AffineMap.create(rank, 0, exprs)
  end

  defp expand_for_output(input_shape, output_shape)
       when tuple_size(output_shape) >= tuple_size(input_shape) do
    output_rank = tuple_size(output_shape)
    rank = tuple_size(input_shape)
    expanded = List.duplicate(1, output_rank - rank) ++ Tuple.to_list(input_shape)
    List.to_tuple(expanded)
  end

  defp gen_indexing_maps(input1_shape, out_shape) do
    [
      expand_for_output(input1_shape, out_shape) |> gen_affine_map(),
      gen_affine_map(out_shape)
    ]
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  defp gen_indexing_maps(
         input1_shape,
         input2_shape,
         out_shape
       ) do
    [
      expand_for_output(input1_shape, out_shape) |> gen_affine_map(),
      expand_for_output(input2_shape, out_shape) |> gen_affine_map(),
      gen_affine_map(out_shape)
    ]
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  defp gen_iterator_types({}, {}) do
    ~a{[]}
  end

  defp gen_iterator_types({_}, {_}) do
    ~a{["parallel"]}
  end

  defp gen_iterator_types(_input1, _input2, _output) do
    ~a{["parallel", "parallel"]}
  end

  defp gen_expand(
         _env,
         value,
         %{shape: in_shape},
         %{shape: out_shape}
       )
       when tuple_size(in_shape) == tuple_size(out_shape) do
    value
  end

  defp gen_expand(
         %Env{block: block},
         value,
         %{type: type, shape: in_shape} = input_t,
         %{shape: out_shape} = _output_t
       ) do
    mlir block: block do
      shape = expand_for_output(in_shape, out_shape)
      t = %{input_t | type: type, shape: shape}
      rank_diff = tuple_size(out_shape) - tuple_size(in_shape)

      pairs =
        Range.new(0, tuple_size(in_shape) - 1, 1)
        |> Enum.map(fn i -> [i, i + rank_diff] end)

      Tensor.expand_shape(value, reassociation: Tensor.reassociation(pairs)) >>> gen_type(t)
    end
  end

  def gen_op(%Env{block: block}, %Nx.Tensor{
        data: %Nx.Defn.Expr{op: :parameter, args: [pos]}
      })
      when is_integer(pos) do
    arg = block |> Beaver.MLIR.Block.get_arg!(pos)

    arg_cnt = Beaver.Walker.arguments(block) |> Enum.count()

    if pos >= arg_cnt do
      raise "arg ##{pos} out of bound, arg_cnt: #{arg_cnt}"
    end

    if MLIR.is_null(arg) do
      raise "arg ##{pos} not found"
    end

    arg
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
        gen_type(t)
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
          shape: {}
        } = t
      )
      when is_integer(value) or is_float(value) do
    mlir block: block do
      t_str = gen_type(t) |> MLIR.to_string()

      TOSA.const({:value, ~a{dense<#{value}> : #{t_str}}}) >>>
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
      t_str = gen_type(t) |> MLIR.to_string()

      Arith.constant({:value, ~a[dense<(#{re}, #{im})> : #{t_str}]}) >>>
        gen_type(t)
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

  # unary tosa
  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: op, args: [input1]}} = t
      )
      when op in [
             :negate,
             :abs,
             :bitwise_not,
             :exp,
             :logical_not,
             :log,
             :tanh,
             :rsqrt,
             :is_nan,
             :is_infinity,
             :sigmoid
           ] do
    mlir block: block do
      input1_value = gen_op(env, input1)
      input1_value = TOSA.cast(input1_value) >>> gen_type(%{input1 | type: t.type})

      case op do
        :negate ->
          TOSA.negate(input1_value) >>> gen_type(t)

        :abs ->
          TOSA.abs(input1_value) >>> gen_type(t)

        :bitwise_not ->
          TOSA.bitwise_not(input1_value) >>> gen_type(t)

        :logical_not ->
          input1_value = TOSA.cast(input1_value) >>> gen_type(%{t | type: {:u, 1}})
          result = TOSA.logical_not(input1_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(result) >>> gen_type(t)

        :exp ->
          TOSA.exp(input1_value) >>> gen_type(t)

        :log ->
          TOSA.log(input1_value) >>> gen_type(t)

        :tanh ->
          TOSA.tanh(input1_value) >>> gen_type(t)

        :rsqrt ->
          TOSA.rsqrt(input1_value) >>> gen_type(t)

        :sigmoid ->
          TOSA.sigmoid(input1_value) >>> gen_type(t)

        :is_nan ->
          # Arith.cmpf(input1_value, input1_value, predicate: Arith.cmp_f_predicate(:uno)) >>>
          #   gen_type(t)
          c = TOSA.equal(input1_value, input1_value) >>> gen_type(%{t | type: {:u, 1}})
          c = TOSA.logical_not(c) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :is_infinity ->
          input1_value = gen_op(env, input1)
          input1_type_str = gen_type(input1) |> MLIR.to_string()

          inf =
            TOSA.const({:value, ~a{dense<0x7F800000> : #{input1_type_str}}}) >>> gen_type(input1)

          abs = TOSA.abs(input1_value) >>> gen_type(input1)
          equal = TOSA.equal(inf, abs) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(equal) >>> gen_type(t)
      end
    end
  end

  def gen_op(
        env,
        %Nx.Tensor{shape: {}, data: %Nx.Defn.Expr{op: :all, args: [%{shape: {}} = input1, _]}}
      ) do
    gen_op(env, input1)
  end

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{
          data: %Nx.Defn.Expr{
            op: :all,
            args: [%{shape: in_shape} = input1, [axes: axes, keep_axes: keep_axes]]
          }
        } = t
      )
      when is_list(axes) do
    mlir block: block do
      input1 = gen_op(env, input1)
      input1 = TOSA.cast(input1) >>> gen_type(%{t | shape: in_shape, type: {:u, 1}})

      {in_shape, mlir_value} =
        Enum.reduce(
          axes,
          {Tuple.to_list(in_shape), input1},
          fn axis, {in_shape, mlir_value} ->
            out_shape = List.replace_at(in_shape, axis, 1)

            reduced =
              TOSA.reduce_all(mlir_value, axis: Attribute.integer(Type.i64(), axis)) >>>
                gen_type(%{t | shape: List.to_tuple(out_shape), type: {:u, 1}})

            {out_shape, reduced}
          end
        )

      mlir_value = TOSA.cast(mlir_value) >>> gen_type(%{t | shape: List.to_tuple(in_shape)})

      if keep_axes do
        mlir_value
      else
        Tensor.collapse_shape(mlir_value, reassociation: Tensor.reassociation([])) >>> gen_type(t)
      end
    end
  end

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{
          data:
            %Nx.Defn.Expr{
              op: :all,
              args: [%{shape: in_shape} = input1, [axes: nil, keep_axes: keep_axes]]
            } = expr
        } = t
      ) do
    # if axes is nil, replace it with a list of every axis
    mlir block: block do
      rank = tuple_size(in_shape)
      axes = Range.new(0, rank - 1, 1) |> Enum.to_list()

      expr = %{
        expr
        | args: [input1, [axes: axes, keep_axes: keep_axes]]
      }

      gen_op(env, %{t | data: expr})
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
            MemRef.store(conjugate_element, conjugate_memref, index) >>> []
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

  # unary linalg
  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{type: type, data: %Nx.Defn.Expr{op: op, args: [input]}} = t
      )
      when op in [
             :population_count,
             :count_leading_zeros,
             :cos,
             :sin,
             :sqrt,
             :tan,
             :erf,
             :cbrt,
             :expm1,
             :log1p
           ] do
    mlir block: block do
      input_value = gen_op(env, input)
      input_value = TOSA.cast(input_value) >>> gen_type(t)

      out_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0])) >>>
          gen_type(t)

      Linalg.generic [
        input_value,
        out_tensor,
        operand_segment_sizes: ODS.operand_segment_sizes([1, 1]),
        indexing_maps: gen_indexing_maps(input.shape, t.shape),
        iterator_types: gen_iterator_types(input.shape, t.shape)
      ] do
        region do
          block bb0(arg0 >>> gen_type(type), out >>> gen_type(type)) do
            %MLIR.Value{} = out

            result =
              case op do
                :population_count ->
                  Math.ctpop(arg0) >>> gen_type(type)

                :count_leading_zeros ->
                  Math.ctlz(arg0) >>> gen_type(type)

                :cos ->
                  Math.cos(arg0) >>> gen_type(type)

                :sin ->
                  Math.sin(arg0) >>> gen_type(type)

                :sqrt ->
                  Math.sqrt(arg0) >>> gen_type(type)

                :tan ->
                  Math.tan(arg0) >>> gen_type(type)

                :erf ->
                  Math.erf(arg0) >>> gen_type(type)

                :cbrt ->
                  abs = Math.abs(arg0) >>> gen_type(type)

                  third =
                    Arith.constant(value: Attribute.float(gen_type(type), 0.333333343)) >>>
                      gen_type(type)

                  pow = Math.powf(abs, third) >>> gen_type(type)
                  Math.copysign(pow, arg0) >>> gen_type(type)

                :expm1 ->
                  Math.expm1(arg0) >>> gen_type(type)

                :log1p ->
                  Math.log1p(arg0) >>> gen_type(type)
              end

            Linalg.yield(result) >>> []
          end
        end
      end >>> gen_type(t)
    end
  end

  # binary linalg
  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{type: type, data: %Nx.Defn.Expr{op: op, args: [a, b]}} = t
      )
      when op in [:remainder, :atan2] do
    mlir block: block do
      a_value = gen_op(env, a)
      a_value = gen_expand(env, a_value, a, t)
      b_value = gen_op(env, b)
      b_value = gen_expand(env, b_value, b, t)

      out_tensor =
        Bufferization.alloc_tensor(operand_segment_sizes: ODS.operand_segment_sizes([0, 0])) >>>
          gen_type(t)

      Linalg.generic [
        a_value,
        b_value,
        out_tensor,
        operand_segment_sizes: ODS.operand_segment_sizes([2, 1]),
        indexing_maps: gen_indexing_maps(a.shape, b.shape, t.shape),
        iterator_types: gen_iterator_types(a.shape, b.shape, t.shape)
      ] do
        region do
          block bb0(arg0 >>> gen_type(type), arg1 >>> gen_type(type), out >>> gen_type(type)) do
            %MLIR.Value{} = out

            result =
              case op do
                :remainder ->
                  case type do
                    {:f, _} ->
                      Arith.remf(arg0, arg1) >>> gen_type(type)

                    {:i, _} ->
                      Arith.remui(arg0, arg1) >>> gen_type(type)

                    {:s, _} ->
                      Arith.remsi(arg0, arg1) >>> gen_type(type)
                  end

                :atan2 ->
                  Math.atan2(arg0, arg1) >>> gen_type(type)
              end

            Linalg.yield(result) >>> []
          end
        end
      end >>> gen_type(t)
    end
  end

  def gen_op(env, %Nx.Tensor{
        data: %Nx.Defn.Expr{
          op: :optional,
          args:
            [
              %{
                data: %{op: :logical_not}
              },
              %{
                data: %{op: :equal}
              }
            ] = list
        }
      }) do
    gen_op(env, List.first(list))
  end

  # binary tosa
  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: op, args: [a, b]}} = t
      ) do
    mlir block: block do
      a_t = %{a | type: t.type} |> gen_type
      b_t = %{b | type: t.type} |> gen_type
      a_value = gen_op(env, a)
      b_value = gen_op(env, b)

      {a_value, b_value} =
        case op do
          _ when op in [:equal] ->
            b_value =
              if a.type != b.type do
                TOSA.cast(b_value) >>> gen_type(%{b | type: a.type})
              else
                b_value
              end

            {a_value, b_value}

          _ when op in [:logical_or, :logical_xor, :logical_and] ->
            a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: {:u, 1}})
            b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: {:u, 1}})
            {a_value, b_value}

          _ ->
            a_value = TOSA.cast(a_value) >>> a_t
            b_value = TOSA.cast(b_value) >>> b_t
            {a_value, b_value}
        end

      case op do
        :subtract ->
          TOSA.sub(a_value, b_value) >>> gen_type(t)

        :less_equal ->
          c = TOSA.greater_equal(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :greater_equal ->
          c = TOSA.greater_equal(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :less ->
          c = TOSA.greater(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :greater ->
          c = TOSA.greater(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :equal ->
          c = TOSA.equal(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :not_equal ->
          c = TOSA.equal(b_value, a_value) >>> gen_type(%{t | type: {:u, 1}})
          c = TOSA.logical_not(c) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :logical_and ->
          c = TOSA.logical_and(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :logical_or ->
          c = TOSA.logical_or(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :logical_xor ->
          c = TOSA.logical_xor(a_value, b_value) >>> gen_type(%{t | type: {:u, 1}})
          TOSA.cast(c) >>> gen_type(t)

        :add ->
          TOSA.add(a_value, b_value) >>> gen_type(t)

        :max ->
          TOSA.maximum(a_value, b_value) >>> gen_type(t)

        :min ->
          TOSA.minimum(a_value, b_value) >>> gen_type(t)

        :bitwise_and ->
          TOSA.bitwise_and(a_value, b_value) >>> gen_type(t)

        :bitwise_or ->
          TOSA.bitwise_or(a_value, b_value) >>> gen_type(t)

        :left_shift ->
          TOSA.logical_left_shift(a_value, b_value) >>> gen_type(t)

        :right_shift ->
          case t.type do
            {:u, _} ->
              TOSA.logical_right_shift(a_value, b_value) >>> gen_type(t)

            {:s, _} ->
              TOSA.arithmetic_right_shift(a_value, b_value, round: Attribute.bool(false)) >>>
                gen_type(t)
          end

        :multiply ->
          TOSA.mul(a_value, b_value, shift: Attribute.integer(Type.i32(), 0)) >>> gen_type(t)

        :divide ->
          b_r = TOSA.reciprocal(b_value) >>> b_t
          TOSA.mul(a_value, b_r, shift: Attribute.integer(Type.i(32), 0)) >>> gen_type(t)

        :quotient ->
          a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: {:u, 32}})
          b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: {:u, 32}})
          result = TOSA.div(a_value, b_value) >>> gen_type(%{t | type: {:u, 32}})
          TOSA.cast(result) >>> gen_type(t)

        :power ->
          {_, width} = a.type
          width = min(width, 32)
          a_value = TOSA.cast(a_value) >>> gen_type(%{a | type: {:f, width}})
          b_value = TOSA.cast(b_value) >>> gen_type(%{b | type: {:f, width}})
          result = TOSA.pow(a_value, b_value) >>> gen_type(%{t | type: {:f, width}})
          TOSA.cast(result) >>> gen_type(t)

        _ ->
          raise "Unsupported binary op: #{inspect(t, structs: false, pretty: true)}"
      end
    end
  end

  def gen_op(
        %Env{block: block} = env,
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :select, args: [pred, on_true, on_false]}} = t
      ) do
    mlir block: block do
      pred_value = gen_op(env, pred)
      pred_t = %{pred | type: {:u, 1}}
      pred_value = TOSA.cast(pred_value) >>> gen_type(pred_t)
      pred_value = gen_expand(env, pred_value, pred_t, t)
      on_true_value = gen_op(env, on_true)
      on_false_value = gen_op(env, on_false)
      on_true_value = TOSA.cast(on_true_value) >>> gen_type(%{on_true | type: t.type})
      on_false_value = TOSA.cast(on_false_value) >>> gen_type(%{on_false | type: t.type})
      TOSA.select(pred_value, on_true_value, on_false_value) >>> gen_type(t)
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
