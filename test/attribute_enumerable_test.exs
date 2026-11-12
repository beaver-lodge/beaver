defmodule AttributeEnumerableTest do
  @moduledoc "Tests the Enumerable protocol for MLIR attributes across element representations."
  use Beaver.Case, async: true
  alias Beaver.{MLIR, Native}
  alias MLIR.{Attribute, CAPI, NamedAttribute, Type}

  @elements [
    {:i32, "Int32", [1, 2, 3, 4, 5]},
    {:i1, "Bool", [true, false, true]},
    {:i8, "Int8", [-1, 0, 1]},
    {:ui8, "UInt8", [0, 128, 255]},
    {:f32, "Float", [1.0, 2.0, 3.0, 4.0, 5.0]},
    {:f64, "Double", [1.0, 2.0, 3.0]},
    {:i16, "Int16", [-100, 0, 100]},
    {:ui16, "UInt16", [0, 32_768, 65_535]},
    {:i64, "Int64", [-10_000_000_000, 0, 10_000_000_000]},
    {:ui64, "UInt64", [0, 10_000_000_000, 18_446_744_073_709_551_615]},
    {:index, "Index", [0, 1, 2]}
  ]
  for {type, accessor, values} <- @elements do
    test "dense #{type} elements", %{ctx: ctx} do
      values = unquote(values)
      tensor = Type.ranked_tensor!([length(values)], apply(Type, unquote(type), [[ctx: ctx]]))
      attr = Attribute.dense_elements(values, tensor, ctx: ctx)
      assert_sequence(attr, values)
      assert_native_elements(attr, values, unquote(accessor))
      if unquote(type) in [:i32, :f32], do: assert_numeric_sequence(attr, values)
    end
  end

  test "opaque string and empty elements", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    values = ["hello", "world", "!"]
    type = Type.opaque("test", "str", ctx: ctx)
    attr = Attribute.dense_elements(values, Type.ranked_tensor!([3], type), ctx: ctx)
    assert_sequence(attr, values)

    for {v, i} <- Enum.with_index(values),
        do: assert(MLIR.to_string(CAPI.mlirDenseElementsAttrGetStringValue(attr, i)) == v)

    empty_type = Type.ranked_tensor!([0], Type.opaque("test2", "str2", ctx: ctx))
    assert Enum.empty?(Attribute.dense_elements([], empty_type, ctx: ctx))
  end

  test "arrays", %{ctx: ctx} do
    integer = &Attribute.integer(Type.i32(ctx: ctx), &1)
    attr = Attribute.array(Enum.map(0..5, integer), ctx: ctx)
    assert Enum.count(attr) == 6
    assert Enum.member?(attr, integer.(3))
    refute Enum.member?(attr, integer.(99))
    assert length(Enum.to_list(attr)) == 6
    assert Enum.reduce(attr, 0, fn _, n -> n + 1 end) == 6
    assert MLIR.equal?(integer.(2), Enum.at(attr, 2))
  end

  for {native, values} <- [{Native.I32, [1, 2, 3, 4, 5]}, {Native.F32, [1.0, 2.0, 3.0, 4.0, 5.0]}] do
    test "dense array #{native}", %{ctx: ctx} do
      values = unquote(values)
      attr = Attribute.dense_array(values, unquote(native), ctx: ctx)
      assert_sequence(attr, values)
      assert_numeric_sequence(attr, values)
      assert Enum.empty?(Attribute.dense_array([], unquote(native), ctx: ctx))
    end
  end

  @splats [
    {:i1, "Bool", true, 5},
    {:i1, "Bool", false, 5},
    {:i8, "Int8", 42, 3},
    {:ui8, "UInt8", 255, 3},
    {:i32, "Int32", 100_000, 4},
    {:ui32, "UInt32", 4_000_000_000, 4},
    {:i64, "Int64", -10_000_000_000, 2},
    {:ui64, "UInt64", 18_000_000_000, 2},
    {:f32, "Float", 0.5, 3},
    {:f64, "Double", 2.71828, 3}
  ]
  for {type, accessor, value, count} <- @splats do
    @tag splat: value
    test "#{type} splat #{value}", %{ctx: ctx, splat: value} do
      type = apply(Type, unquote(type), [[ctx: ctx]])

      attr = Attribute.dense_elements(unquote(value), Type.ranked_tensor!([unquote(count)], type), ctx: ctx)

      values = List.duplicate(unquote(value), unquote(count))
      assert_sequence(attr, values)
      assert Enum.all?(attr, &(&1 == unquote(value)))
      assert_native_elements(attr, values, unquote(accessor))
      if is_boolean(value), do: refute(Enum.member?(attr, not value))
    end
  end

  test "attribute splat", %{ctx: ctx} do
    tensor = Type.ranked_tensor!([3], Type.f64(ctx: ctx))
    attr = Attribute.dense_elements(Attribute.float(Type.f64(), 2.71828), tensor, ctx: ctx)
    assert_sequence(attr, List.duplicate(2.71828, 3))
    assert_native_elements(attr, List.duplicate(2.71828, 3), "Double")
  end

  test "dictionaries", %{ctx: ctx} do
    named = &NamedAttribute.get("key#{&1}", Attribute.integer(Type.i32(ctx: ctx), &1), ctx: ctx)
    dict = Attribute.dictionary([named.(1), named.(2)], ctx: ctx)
    assert Enum.count(dict) == 2
    assert Enum.member?(dict, named.(1))
    refute Enum.member?(dict, named.(3))
    assert Enumerable.count(dict) == {:ok, 2}
    assert Enumerable.member?(dict, named.(2)) == {:ok, true}
    assert Enum.reduce(dict, 0, &(Attribute.value(NamedAttribute.attribute(&1)) + &2)) == 3
    assert Enum.count(Enum.slice(dict, 0..1)) == 2
    assert {:ok, 2, _} = Enumerable.slice(dict)
  end

  defp assert_sequence(attr, values) do
    assert Enum.count(attr) == length(values)
    assert Enum.to_list(attr) == values

    for {value, index} <- Enum.with_index(values) do
      assert Enum.member?(attr, value)
      assert Enum.at(attr, index) == value
    end
  end

  defp assert_numeric_sequence(attr, values) do
    refute Enum.member?(attr, if(is_float(hd(values)), do: 99.0, else: 99))
    assert Enum.reduce(attr, 0, &+/2) == 15
    assert Enum.slice(attr, 1..3) == Enum.slice(values, 1..3)
  end

  defp assert_native_elements(attr, values, accessor) do
    for {value, index} <- Enum.with_index(values),
        do: assert(Native.to_term(apply(CAPI, :"mlirDenseElementsAttrGet#{accessor}Value", [attr, index])) == value)
  end
end
