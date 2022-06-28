defmodule Beaver.Defn.ExprTest do
  # TODO: running this in async will trigger multi-thread check in MLIR and crash
  use ExUnit.Case, async: false
  import Nx, only: :sigils
  import Nx.Defn
  alias Beaver.Nx.Assert
  require Assert

  setup do
    Nx.Defn.default_options(compiler: Beaver.Nx.Compiler)
    :ok
  end

  defp evaluate(fun, args) do
    fun |> Nx.Defn.jit(compiler: Nx.Defn.Evaluator) |> apply(args)
  end

  describe "tuples" do
    defn(add_subtract_tuple(a, b), do: {a + b, a - b})

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
    defn(constants, do: @two)
    defn(add_two_attribute(t), do: t + @two)

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn(add_2x2_attribute(t), do: t + @two_per_two)
    defn(add_2x2_constant(), do: @two_per_two + @two_per_two)
    defn(add_2x2_constant(_), do: @two_per_two + @two_per_two)

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
    defn(infinity, do: Nx.Constants.infinity())
    defn(neg_infinity, do: Nx.Constants.neg_infinity())
    defn(nan, do: Nx.Constants.nan())

    test "handles non-finite constants correctly" do
      Assert.equal(infinity(), Nx.Constants.infinity())
      Assert.equal(neg_infinity(), Nx.Constants.neg_infinity())
      Assert.equal(nan(), Nx.Constants.nan())
    end

    defn(negate_infinity, do: Nx.negate(Nx.Constants.infinity()))
    defn(negate_neg_infinity, do: Nx.negate(Nx.Constants.infinity()))

    test "sanity check constants" do
      Assert.equal(negate_infinity(), Nx.Constants.neg_infinity())
      Assert.equal(infinity(), Nx.Constants.infinity())
    end
  end

  describe "float16" do
    defn(return_float, do: Nx.tensor(1, type: {:f, 16}))

    test "supports float16 return types" do
      Assert.equal(return_float(), Nx.tensor(1, type: {:f, 16}))
    end
  end
end
