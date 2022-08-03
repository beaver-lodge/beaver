defmodule Beaver.Defn.ExprTest do
  # TODO: running this in async will trigger multi-thread check in MLIR and crash
  use ExUnit.Case, async: true
  # import Nx, only: :sigils
  import Nx.Defn
  alias Manx.Assert
  import Manx.Assert
  require Assert

  @moduletag :nx
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler)
    :ok
  end

  defp evaluate(fun, args) do
    Nx.Defn.jit(fun, args, compiler: Nx.Defn.Evaluator)
  end

  describe "tuples" do
    defn add_subtract_tuple(a, b), do: {a + b, a - b}

    test "on results" do
      Assert.equal(add_subtract_tuple(2, 3), {Nx.tensor(5), Nx.tensor(-1)})

      Assert.equal(
        add_subtract_tuple(Nx.tensor([-1, 0, 1]), Nx.tensor([10, 10, 10])),
        {Nx.tensor([9, 10, 11]), Nx.tensor([-11, -10, -9])}
      )

      Assert.equal(
        add_subtract_tuple(Nx.tensor([-1, 0, 1]), 10),
        {Nx.tensor([9, 10, 11]), Nx.tensor([-11, -10, -9])}
      )
    end

    defn(pattern_tuple({a, b}), do: a + b)

    test "on patterns" do
      Assert.equal(pattern_tuple({2, 3}), Nx.tensor(5))

      Assert.equal(
        pattern_tuple({Nx.tensor([1, 2]), Nx.tensor([[3], [4]])}),
        Nx.tensor([[4, 5], [5, 6]])
      )
    end

    defn(calls_pattern_tuple(a, b), do: pattern_tuple({a, b}))

    test "on inlined tuples" do
      Assert.equal(calls_pattern_tuple(2, 3), Nx.tensor(5))

      Assert.equal(
        calls_pattern_tuple(Nx.tensor([1, 2]), Nx.tensor([[3], [4]])),
        Nx.tensor([[4, 5], [5, 6]])
      )
    end
  end

  describe "tensor constants" do
    @two 2
    defn constants, do: @two
    defn add_two_attribute(t), do: t + @two

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn add_2x2_attribute(t), do: t + @two_per_two
    defn add_2x2_constant(), do: @two_per_two + @two_per_two
    defn add_2x2_constant(_), do: @two_per_two + @two_per_two

    test "handles tensors as constants" do
      Assert.equal(constants(), Nx.tensor(2))
    end

    test "expands module attributes to scalars" do
      Assert.equal(add_two_attribute(1), Nx.tensor(3))
      Assert.equal(add_two_attribute(Nx.tensor([1, 2, 3])), Nx.tensor([3, 4, 5]))
    end

    test "expands module attributes to tensors" do
      Assert.equal(add_2x2_attribute(1), Nx.tensor([[2, 3], [4, 5]]))
      Assert.equal(add_2x2_attribute(Nx.tensor([1, 2])), Nx.tensor([[2, 4], [4, 6]]))
    end

    test "constants should be folded" do
      Assert.equal(add_2x2_constant(), Nx.tensor([[2, 4], [6, 8]]))
      Assert.equal(add_2x2_constant(1), Nx.tensor([[2, 4], [6, 8]]))
    end
  end

  describe "non finite" do
    defn infinity, do: Nx.Constants.infinity()
    defn neg_infinity, do: Nx.Constants.neg_infinity()
    defn nan, do: Nx.Constants.nan()

    test "handles non-finite constants correctly" do
      Assert.equal(infinity(), Nx.Constants.infinity())
      Assert.equal(neg_infinity(), Nx.Constants.neg_infinity())
      Assert.equal(nan(), Nx.Constants.nan())
    end

    defn negate_infinity, do: Nx.negate(Nx.Constants.infinity())
    defn negate_neg_infinity, do: Nx.negate(Nx.Constants.infinity())

    test "sanity check constants" do
      Assert.equal(negate_infinity(), Nx.Constants.neg_infinity())
      Assert.equal(infinity(), Nx.Constants.infinity())
    end
  end

  describe "float16" do
    defn return_float, do: Nx.tensor(1, type: {:f, 16})

    test "supports float16 return types" do
      Assert.equal(return_float(), Nx.tensor(1, type: {:f, 16}))
    end
  end

  describe "complex" do
    defn return_complex, do: Nx.complex(1, 2)
    defn return_complex_tensor, do: Nx.broadcast(Nx.complex(1, 2), {3, 3, 3})

    test "supports complex return types" do
      Assert.equal(return_complex(), Nx.tensor(Complex.new(1, 2)))
      Assert.equal(return_complex_tensor(), Nx.broadcast(Complex.new(1, 2), {3, 3, 3}))
    end
  end

  describe "conjugate" do
    defn conjugate(x), do: Nx.conjugate(x)

    test "correctly returns complex conjugate" do
      assert_equal(conjugate(Nx.tensor(Complex.new(1, 2))), Nx.tensor(Complex.new(1, -2)))
      # This differs from the Nx doctest, which I believe should also return -0
      assert_equal(conjugate(Nx.tensor(1)), Nx.tensor(Complex.new(1, -0.0)))

      assert_equal(
        conjugate(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)])),
        Nx.tensor([Complex.new(1, -2), Complex.new(2, 4)])
      )
    end
  end

  describe "imag" do
    defn imag(x), do: Nx.imag(x)

    test "correctly returns imaginary part of complex" do
      assert_equal(imag(Nx.tensor(Complex.new(1, 2))), Nx.tensor(2.0))
      assert_equal(imag(Nx.tensor(1)), Nx.tensor(0.0))

      assert_equal(
        imag(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)])),
        Nx.tensor([2.0, -4.0])
      )
    end
  end

  describe "+/2" do
    @describetag :plus
    defn add_two(a, b), do: a + b

    test "same shape and type" do
      assert_equal(add_two(1.0, 2.0), Nx.tensor(3.0))
      assert_equal(add_two(1, 2), Nx.tensor(3))

      assert_equal(add_two(Nx.tensor([1, 2]), Nx.tensor([3, 4])), Nx.tensor([4, 6]))
      assert_equal(add_two(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])), Nx.tensor([4.0, 6.0]))
    end

    test "different types" do
      tensors = [
        {1, 2},
        {1.0, 2},
        {1.0, 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), 3},
        {Nx.tensor([1, 2], type: {:u, 8}), -3},
        {Nx.tensor([1, 2], type: {:u, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:s, 8}), 3},
        {Nx.tensor([1, 2], type: {:s, 8}), 3.0},
        {Nx.tensor([1, 2], type: {:f, 32}), 3},
        {Nx.tensor([1, 2], type: {:f, 32}), 3.0},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(-3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:u, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3, type: {:s, 16})},
        {Nx.tensor([1, 2], type: {:s, 8}), Nx.tensor(3.0, type: {:f, 32})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:u, 16})},
        {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3, type: {:s, 16})}
        # {Nx.tensor([1, 2], type: {:f, 32}), Nx.tensor(3.0, type: {:f, 64})}
      ]

      for {left, right} <- tensors do
        assert_all_close(add_two(left, right), evaluate(&add_two/2, [left, right]))
        assert_all_close(add_two(right, left), evaluate(&add_two/2, [right, left]))
      end
    end

    defn add_two_int(t), do: t + 2
    defn add_two_float(t), do: t + 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 16}),
        Nx.tensor([1, 2], type: {:u, 32}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
        # Nx.tensor([1, 2], type: {:f, 64})
      ]

      for t <- tensors do
        assert_equal(add_two_int(t), Nx.add(t, 2))
        assert_equal(add_two_float(t), Nx.add(t, 2.0))
      end
    end

    test "broadcast" do
      tensors = [
        {Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([1, 2]), Nx.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]])},
        {Nx.tensor([[[10], [20]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[100], [200], [300]]]),
         Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([1, 2])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[1, 2]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[[1, 2]]], [[[3, 4]]]])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[[1, 2]], [[3, 4]]])}
      ]

      for {left, right} <- tensors do
        assert_all_close(add_two(left, right), evaluate(&add_two/2, [left, right]))
        assert_all_close(add_two(right, left), evaluate(&add_two/2, [right, left]))
      end
    end

    test "names" do
      left = Nx.tensor([[10, 20]], names: [nil, :tens])
      right = Nx.tensor([[1], [2]], names: [:ones, nil])
      assert add_two(left, right).names == [:ones, :tens]
    end
  end

  describe "//2" do
    defn divide_two(a, b), do: a / b

    test "parameters" do
      tensors = [
        {1, 2},
        {1, Nx.tensor([1.0, 2.0, 3.0])},
        {Nx.tensor([1, 2, 3]), 1},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
        {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
      ]

      for {left, right} <- tensors do
        assert_all_close(divide_two(left, right), Nx.divide(left, right))
        assert_all_close(divide_two(right, left), Nx.divide(right, left))
      end
    end

    defn divide_two_int(t), do: t / 2
    defn divide_two_float(t), do: t / 2.0

    test "constants" do
      tensors = [
        Nx.tensor([1, 2], type: {:u, 8}),
        Nx.tensor([1, 2], type: {:u, 16}),
        Nx.tensor([1, 2], type: {:u, 32}),
        Nx.tensor([1, 2], type: {:s, 8}),
        Nx.tensor([1, 2], type: {:s, 32}),
        Nx.tensor([1, 2], type: {:f, 32})
        # Nx.tensor([1, 2], type: {:f, 64})
      ]

      for t <- tensors do
        assert_all_close(divide_two_int(t), Nx.divide(t, 2))
        assert_all_close(divide_two_float(t), Nx.divide(t, 2.0))
      end
    end
  end

  describe "remainder" do
    defn remainder(a, b), do: Nx.remainder(a, b)

    test "integers" do
      left = Nx.tensor([-1023, 1023])
      right = Nx.tensor([[-4], [4]])
      assert Nx.shape(remainder(left, right)) == {2, 2}
      assert_all_close(remainder(left, right), Nx.remainder(left, right))
    end

    test "floats" do
      left = Nx.tensor([-8.3, -8.4, -8.5, 8.3, 8.4, 8.5])
      right = Nx.tensor([[-4.2], [-4.1], [-4.0], [4.0], [4.1], [4.2]])
      assert Nx.shape(remainder(left, right)) == {6, 6}
      assert_all_close(remainder(left, right), Nx.remainder(left, right))
    end
  end

  describe "element-wise arith operators" do
    @tensors [
      {1, 2},
      {1, Nx.tensor([1.0, 2.0, 3.0])},
      {Nx.tensor([1, 2, 3]), 1},
      {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
      {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
      {Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32})}
    ]

    defn subtract_two(a, b), do: a - b

    test "-" do
      for {left, right} <- @tensors do
        assert_all_close(subtract_two(left, right), Nx.subtract(left, right))
        assert_all_close(subtract_two(right, left), Nx.subtract(right, left))
      end
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      for {left, right} <- @tensors do
        assert_all_close(multiply_two(left, right), Nx.multiply(left, right))
        assert_all_close(multiply_two(right, left), Nx.multiply(right, left))
      end
    end

    defn unary_minus(a), do: -a

    test "negate" do
      for t <- [
            Nx.tensor([-1, 0, 1], type: {:u, 8}),
            Nx.tensor([-1, 0, 1]),
            Nx.tensor([-1.0, 1.0])
          ] do
        assert_equal(unary_minus(t), Nx.negate(t))
      end
    end

    defn max_two(a, b), do: max(a, b)

    test "max" do
      for {left, right} <- @tensors do
        assert_all_close(max_two(left, right), Nx.max(left, right))
        assert_all_close(max_two(right, left), Nx.max(right, left))
      end
    end

    defn min_two(a, b), do: min(a, b)

    test "min" do
      for {left, right} <- @tensors do
        assert_all_close(min_two(left, right), Nx.min(left, right))
        assert_all_close(min_two(right, left), Nx.min(right, left))
      end
    end

    defn power_two(a, b), do: Nx.power(a, b)

    test "power" do
      for {left, right} <- @tensors do
        case left do
          %{type: {_, 8}} ->
            nil

          _ ->
            assert_all_close(power_two(left, right), Nx.power(left, right))
            assert_all_close(power_two(right, left), Nx.power(right, left))
        end
      end
    end

    defn atan2_two(a, b), do: Nx.atan2(a, b)

    test "atan2" do
      <<neg_zero::float>> = <<0x8000000000000000::64>>
      left = Nx.tensor([-1.0, neg_zero, 0.0, 1.0])
      right = Nx.tensor([[-1.0], [neg_zero], [0.0], [1.0]])

      assert_all_close(atan2_two(left, right), Nx.atan2(left, right))
      assert_all_close(atan2_two(right, left), Nx.atan2(right, left))
    end

    defn quotient_two(a, b), do: Nx.quotient(a, b)

    test "quotient" do
      int_tensors = [
        {1, 2},
        {1, Nx.tensor([1, 2, 3])},
        {Nx.tensor([1, 2, 3]), 1},
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8})},
        {Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 32})}
      ]

      for {left, right} <- int_tensors do
        assert_all_close(quotient_two(left, right), Nx.quotient(left, right))
        assert_all_close(quotient_two(right, left), Nx.quotient(right, left))
      end
    end
  end

  describe "element-wise bitwise operators" do
    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[-2], [-1], [0], [1], [2]])

    defn bitwise_and(a, b), do: a &&& b

    test "bitwise_and" do
      assert Nx.shape(bitwise_and(@left, @right)) == {5, 5}
      assert_equal(bitwise_and(@left, @right), Nx.bitwise_and(@left, @right))
    end

    defn bitwise_or(a, b), do: a ||| b

    test "bitwise_or" do
      assert Nx.shape(bitwise_or(@left, @right)) == {5, 5}
      assert_equal(bitwise_or(@left, @right), Nx.bitwise_or(@left, @right))
    end

    defn bitwise_not(a), do: ~~~a

    test "bitwise_not" do
      assert Nx.shape(bitwise_not(@left)) == {5}
      assert_equal(bitwise_not(@left), Nx.bitwise_not(@left))
    end

    defn bitwise_pc(a), do: Nx.population_count(a)

    test "population_count" do
      assert Nx.shape(bitwise_pc(@left)) == {5}
      assert_equal(bitwise_pc(@left), Nx.population_count(@left))
    end

    defn bitwise_clz(a), do: Nx.count_leading_zeros(a)

    test "count_leading_zeros" do
      assert Nx.shape(bitwise_clz(@left)) == {5}
      assert_equal(bitwise_clz(@left), Nx.count_leading_zeros(@left))
    end

    @left Nx.tensor([-2, -1, 0, 1, 2])
    @right Nx.tensor([[0], [1], [2], [3], [4]])

    defn left_shift(a, b), do: a <<< b

    test "left_shift" do
      assert Nx.shape(left_shift(@left, @right)) == {5, 5}
      assert_equal(left_shift(@left, @right), Nx.left_shift(@left, @right))
    end

    @left_signed Nx.tensor([-128, -127, -2, -1, 0, 1, 2, 126, 127], type: {:s, 8})
    @right_signed Nx.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8]], type: {:s, 8})

    @left_unsigned Nx.tensor([0, 1, 2, 253, 254, 255], type: {:u, 8})
    @right_unsigned Nx.tensor([[0], [1], [2], [3], [4], [5]], type: {:u, 8})

    defn right_shift(a, b), do: a >>> b

    test "right_shift" do
      assert Nx.shape(right_shift(@left_signed, @right_signed)) == {9, 9}

      assert_equal(
        right_shift(@left_signed, @right_signed),
        Nx.right_shift(@left_signed, @right_signed)
      )

      assert Nx.shape(right_shift(@left_unsigned, @right_unsigned)) == {6, 6}

      assert_equal(
        right_shift(@left_unsigned, @right_unsigned),
        Nx.right_shift(@left_unsigned, @right_unsigned)
      )
    end
  end
end
