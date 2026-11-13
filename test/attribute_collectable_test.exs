defmodule AttributeCollectableTest do
  @moduledoc "Tests collection into array, dictionary and shaped attributes."
  use Beaver.Case, async: true
  alias Beaver.{MLIR, Native}
  alias MLIR.{Attribute, NamedAttribute, Type}

  test "arrays", %{ctx: ctx} do
    integer = &Attribute.integer(Type.i32(ctx: ctx), &1)
    array = &Attribute.array(Enum.map(&1, integer), ctx: ctx)
    assert MLIR.equal?(Enum.into([1, 2, 3], array.([]), integer), array.([1, 2, 3]))

    assert_raise ArgumentError, "cannot collect into an attribute that is not empty", fn ->
      Enum.into([1, 2, 3], array.([4, 5]), integer)
    end
  end

  for {native, values} <- [{Native.I32, [1, 2, 3]}, {Native.F32, [1.0, 2.0, 3.0]}, {Native.Bool, [true, false, true]}] do
    test "dense array #{native}", %{ctx: ctx} do
      values = unquote(values)
      build = &Attribute.dense_array(&1, unquote(native), ctx: ctx)
      assert_collected(Enum.into(values, build.([]), & &1), build.(values), values)

      assert_raise ArgumentError, "cannot collect into an attribute that is not empty", fn ->
        Enum.into(tl(values), build.([hd(values)]), & &1)
      end
    end
  end

  test "dictionaries", %{ctx: ctx} do
    empty = Attribute.dictionary([], ctx: ctx)
    integer = Attribute.integer(Type.i32(ctx: ctx), 1)
    created = Enum.into([{"key", 1}], empty, fn {k, _} -> NamedAttribute.get(k, integer, ctx: ctx) end)
    assert Enum.count(created) == 1
    assert MLIR.equal?(integer, created["key"])
  end

  for {type, initial, values} <- [{:i32, [-1, -1, -1], [1, 2, 3]}, {:f32, [0.1, 0.2, 0.3], [1.0, 2.0, 3.0]}] do
    test "dense #{type} elements", %{ctx: ctx} do
      tensor = Type.ranked_tensor!([3], apply(Type, unquote(type), [[ctx: ctx]]))
      build = &Attribute.dense_elements(&1, tensor, ctx: ctx)
      result = Enum.into(unquote(values), build.(unquote(initial)), & &1)
      assert_collected(result, build.(unquote(values)), unquote(values))
    end
  end

  defp assert_collected(result, expected, values) do
    assert MLIR.equal?(result, expected)
    assert Enum.count(result) == length(values)
    for {v, i} <- Enum.with_index(values), do: assert(Enum.at(result, i) == v)
  end
end
