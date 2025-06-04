defmodule AttributeEnumerableTest do
  @moduledoc """
  Test the Enumerable protocol implementation for MLIR attributes.
  """
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias MLIR.Type

  test "dense elements", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test i32 dense elements
    dense_i32 =
      [1, 2, 3, 4, 5]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([5], Type.i32(opts)), opts)

    # Test count
    assert Enum.count(dense_i32) == 5

    # Test member?
    assert Enum.member?(dense_i32, 3)
    refute Enum.member?(dense_i32, 99)

    # Test reduce
    sum = Enum.reduce(dense_i32, 0, &(&1 + &2))
    assert sum == 15

    # Test slice
    assert Enum.slice(dense_i32, 1..3) == [2, 3, 4]
    assert Enum.at(dense_i32, 2) == 3

    # Test direct element access via CAPI and Enum.at
    for i <- 0..4 do
      assert i + 1 ==
               MLIR.CAPI.mlirDenseElementsAttrGetInt32Value(dense_i32, i)
               |> Beaver.Native.to_term()
    end

    assert Enum.at(dense_i32, 0) == 1
    assert Enum.at(dense_i32, 1) == 2
    assert Enum.at(dense_i32, 2) == 3
    assert Enum.at(dense_i32, 3) == 4
    assert Enum.at(dense_i32, 4) == 5

    # Test Enum protocol functions
    assert Enum.count(dense_i32) == 5
    assert Enum.member?(dense_i32, 3)
    refute Enum.member?(dense_i32, 99)
    assert Enum.reduce(dense_i32, 0, &(&1 + &2)) == 15
    assert Enum.slice(dense_i32, 1..3) == [2, 3, 4]

    # Test bool dense elements
    dense_bool =
      [true, false, true]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.i1(opts)), opts)

    assert Enum.count(dense_bool) == 3
    assert Enum.member?(dense_bool, true)
    assert Enum.member?(dense_bool, false)
    assert MLIR.CAPI.mlirDenseElementsAttrGetBoolValue(dense_bool, 0) |> Beaver.Native.to_term()
    refute MLIR.CAPI.mlirDenseElementsAttrGetBoolValue(dense_bool, 1) |> Beaver.Native.to_term()
    assert MLIR.CAPI.mlirDenseElementsAttrGetBoolValue(dense_bool, 2) |> Beaver.Native.to_term()
    assert Enum.at(dense_bool, 0) == true
    assert Enum.at(dense_bool, 1) == false
    assert Enum.at(dense_bool, 2) == true

    # Test int8 dense elements
    dense_i8 =
      [-1, 0, 1]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.i8(opts)), opts)

    assert Enum.count(dense_i8) == 3
    assert Enum.member?(dense_i8, 0)

    assert -1 =
             MLIR.CAPI.mlirDenseElementsAttrGetInt8Value(dense_i8, 0) |> Beaver.Native.to_term()

    assert 0 = MLIR.CAPI.mlirDenseElementsAttrGetInt8Value(dense_i8, 1) |> Beaver.Native.to_term()
    assert 1 = MLIR.CAPI.mlirDenseElementsAttrGetInt8Value(dense_i8, 2) |> Beaver.Native.to_term()

    # Test uint8 dense elements
    dense_u8 =
      [0, 128, 255]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.ui8(opts)), opts)

    assert Enum.count(dense_u8) == 3
    assert Enum.member?(dense_u8, 128)

    assert 0 =
             MLIR.CAPI.mlirDenseElementsAttrGetUInt8Value(dense_u8, 0) |> Beaver.Native.to_term()

    assert 128 =
             MLIR.CAPI.mlirDenseElementsAttrGetUInt8Value(dense_u8, 1) |> Beaver.Native.to_term()

    assert 255 =
             MLIR.CAPI.mlirDenseElementsAttrGetUInt8Value(dense_u8, 2) |> Beaver.Native.to_term()

    # Test f32 dense elements
    dense_f32 =
      [1.0, 2.0, 3.0, 4.0, 5.0]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([5], Type.f32(opts)), opts)

    # Test count
    assert Enum.count(dense_f32) == 5

    # Test member?
    assert Enum.member?(dense_f32, 3.0)
    refute Enum.member?(dense_f32, 99.0)

    # Test reduce
    sum_f = Enum.reduce(dense_f32, 0.0, &(&1 + &2))
    assert sum_f == 15.0

    # Test slice
    assert Enum.slice(dense_f32, 1..3) == [2.0, 3.0, 4.0]
    assert Enum.at(dense_f32, 2) == 3.0

    # Test direct element access
    assert MLIR.CAPI.mlirDenseElementsAttrGetFloatValue(dense_f32, 0) |> Beaver.Native.to_term() ==
             1.0

    assert MLIR.CAPI.mlirDenseElementsAttrGetFloatValue(dense_f32, 1) |> Beaver.Native.to_term() ==
             2.0

    assert MLIR.CAPI.mlirDenseElementsAttrGetFloatValue(dense_f32, 2) |> Beaver.Native.to_term() ==
             3.0

    # Test f64 dense elements
    dense_f64 =
      [1.0, 2.0, 3.0]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.f64(opts)), opts)

    assert Enum.count(dense_f64) == 3
    assert Enum.member?(dense_f64, 2.0)

    assert MLIR.CAPI.mlirDenseElementsAttrGetDoubleValue(dense_f64, 0) |> Beaver.Native.to_term() ==
             1.0

    assert MLIR.CAPI.mlirDenseElementsAttrGetDoubleValue(dense_f64, 1) |> Beaver.Native.to_term() ==
             2.0

    assert MLIR.CAPI.mlirDenseElementsAttrGetDoubleValue(dense_f64, 2) |> Beaver.Native.to_term() ==
             3.0

    # Test int16 dense elements
    dense_i16 =
      [-100, 0, 100]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.i16(opts)), opts)

    assert Enum.count(dense_i16) == 3
    assert Enum.member?(dense_i16, 0)

    assert -100 =
             MLIR.CAPI.mlirDenseElementsAttrGetInt16Value(dense_i16, 0) |> Beaver.Native.to_term()

    assert 0 =
             MLIR.CAPI.mlirDenseElementsAttrGetInt16Value(dense_i16, 1) |> Beaver.Native.to_term()

    assert 100 =
             MLIR.CAPI.mlirDenseElementsAttrGetInt16Value(dense_i16, 2) |> Beaver.Native.to_term()

    # Test uint16 dense elements
    dense_u16 =
      [0, 32768, 65535]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.ui16(opts)), opts)

    assert Enum.count(dense_u16) == 3
    assert Enum.member?(dense_u16, 32768)

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt16Value(dense_u16, 0) |> Beaver.Native.to_term() ==
             0

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt16Value(dense_u16, 1) |> Beaver.Native.to_term() ==
             32768

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt16Value(dense_u16, 2) |> Beaver.Native.to_term() ==
             65535

    # Test int64 dense elements
    dense_i64 =
      [-10_000_000_000, 0, 10_000_000_000]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.i64(opts)), opts)

    assert Enum.count(dense_i64) == 3
    assert Enum.member?(dense_i64, 0)

    assert MLIR.CAPI.mlirDenseElementsAttrGetInt64Value(dense_i64, 0) |> Beaver.Native.to_term() ==
             -10_000_000_000

    assert MLIR.CAPI.mlirDenseElementsAttrGetInt64Value(dense_i64, 1) |> Beaver.Native.to_term() ==
             0

    assert MLIR.CAPI.mlirDenseElementsAttrGetInt64Value(dense_i64, 2) |> Beaver.Native.to_term() ==
             10_000_000_000

    # Test uint64 dense elements
    dense_u64 =
      [0, 10_000_000_000, 18_446_744_073_709_551_615]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.ui64(opts)), opts)

    assert Enum.count(dense_u64) == 3
    assert Enum.member?(dense_u64, 10_000_000_000)

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt64Value(dense_u64, 0) |> Beaver.Native.to_term() ==
             0

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt64Value(dense_u64, 1) |> Beaver.Native.to_term() ==
             10_000_000_000

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt64Value(dense_u64, 2) |> Beaver.Native.to_term() ==
             18_446_744_073_709_551_615

    # Test index dense elements
    dense_index =
      [0, 1, 2]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.index(opts)), opts)

    assert Enum.count(dense_index) == 3
    assert Enum.member?(dense_index, 1)

    assert MLIR.CAPI.mlirDenseElementsAttrGetIndexValue(dense_index, 0) |> Beaver.Native.to_term() ==
             0

    assert MLIR.CAPI.mlirDenseElementsAttrGetIndexValue(dense_index, 1) |> Beaver.Native.to_term() ==
             1

    assert MLIR.CAPI.mlirDenseElementsAttrGetIndexValue(dense_index, 2) |> Beaver.Native.to_term() ==
             2

    MLIR.Context.allow_unregistered_dialects(ctx)
    # Test string dense elements
    dense_str =
      ["hello", "world", "!"]
      |> MLIR.Attribute.dense_elements(
        Type.ranked_tensor!([3], Type.opaque("test", "str", opts)),
        opts
      )

    assert Enum.count(dense_str) == 3
    assert Enum.member?(dense_str, "world")

    assert "hello" =
             MLIR.CAPI.mlirDenseElementsAttrGetStringValue(dense_str, 0) |> MLIR.to_string()

    assert "world" =
             MLIR.CAPI.mlirDenseElementsAttrGetStringValue(dense_str, 1) |> MLIR.to_string()

    assert "!" = MLIR.CAPI.mlirDenseElementsAttrGetStringValue(dense_str, 2) |> MLIR.to_string()

    # Test empty dense elements
    assert Enum.empty?(
             MLIR.Attribute.dense_elements(
               [],
               Type.ranked_tensor!([0], Type.opaque("test2", "str2", opts)),
               opts
             )
           )
  end

  test "arrays", %{ctx: ctx} do
    opts = [ctx: ctx]

    arr =
      [0, 1, 2, 3, 4, 5]
      |> Enum.map(&MLIR.Attribute.integer(Type.i32(opts), &1))
      |> MLIR.Attribute.array(opts)

    assert Enum.count(arr) == 6
    assert Enum.member?(arr, MLIR.Attribute.integer(Type.i32(opts), 3))
    refute Enum.member?(arr, MLIR.Attribute.integer(Type.i32(opts), 99))
    assert Enum.to_list(arr) |> length() == 6
    assert Enum.reduce(arr, 0, fn _, acc -> acc + 1 end) == 6
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 2), Enum.at(arr, 2))
  end

  test "dense arrays", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test i32 dense array
    dense_i32 =
      [1, 2, 3, 4, 5]
      |> MLIR.Attribute.dense_array(Beaver.Native.I32, opts)

    # Test count
    assert Enum.count(dense_i32) == 5

    # Test member?
    assert Enum.member?(dense_i32, 3)
    refute Enum.member?(dense_i32, 99)

    # Test reduce
    sum = Enum.reduce(dense_i32, 0, &(&1 + &2))
    assert sum == 15

    # Test slice
    assert Enum.slice(dense_i32, 1..3) == [2, 3, 4]
    assert Enum.at(dense_i32, 2) == 3

    # Test f32 dense array
    dense_f32 =
      [1.0, 2.0, 3.0, 4.0, 5.0]
      |> MLIR.Attribute.dense_array(Beaver.Native.F32, opts)

    # Test count
    assert Enum.count(dense_f32) == 5

    # Test member?
    assert Enum.member?(dense_f32, 3.0)
    refute Enum.member?(dense_f32, 99.0)

    # Test reduce
    sum_f = Enum.reduce(dense_f32, 0.0, &(&1 + &2))
    assert sum_f == 15.0

    # Test slice
    assert Enum.slice(dense_f32, 1..3) == [2.0, 3.0, 4.0]
    assert Enum.at(dense_f32, 2) == 3.0

    # Test empty array
    assert Enum.empty?(MLIR.Attribute.dense_array([], Beaver.Native.I32, opts))
  end

  test "dense elements splat", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test bool splat
    dense_bool_splat_true =
      true
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([5], Type.i1(opts)), opts)

    dense_bool_splat_false =
      false
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([5], Type.i1(opts)), opts)

    assert Enum.count(dense_bool_splat_true) == 5
    assert Enum.member?(dense_bool_splat_true, true)
    refute Enum.member?(dense_bool_splat_true, false)
    assert Enum.all?(dense_bool_splat_true, &(&1 == true))
    assert Enum.all?(dense_bool_splat_false, &(&1 == false))

    # Test i8 splat
    dense_i8_splat =
      42
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.i8(opts)), opts)

    assert Enum.count(dense_i8_splat) == 3
    assert Enum.member?(dense_i8_splat, 42)
    assert Enum.all?(dense_i8_splat, &(&1 == 42))

    assert MLIR.CAPI.mlirDenseElementsAttrGetInt8Value(dense_i8_splat, 0)
           |> Beaver.Native.to_term() == 42

    # Test u8 splat
    dense_u8_splat =
      255
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.ui8(opts)), opts)

    assert Enum.count(dense_u8_splat) == 3
    assert Enum.member?(dense_u8_splat, 255)

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt8Value(dense_u8_splat, 0)
           |> Beaver.Native.to_term() == 255

    # Test i32 splat
    dense_i32_splat =
      100_000
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([4], Type.i32(opts)), opts)

    assert Enum.count(dense_i32_splat) == 4
    assert Enum.member?(dense_i32_splat, 100_000)

    assert MLIR.CAPI.mlirDenseElementsAttrGetInt32Value(dense_i32_splat, 0)
           |> Beaver.Native.to_term() == 100_000

    # Test u32 splat
    dense_u32_splat =
      4_000_000_000
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([4], Type.ui32(opts)), opts)

    assert Enum.count(dense_u32_splat) == 4
    assert Enum.member?(dense_u32_splat, 4_000_000_000)

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt32Value(dense_u32_splat, 0)
           |> Beaver.Native.to_term() == 4_000_000_000

    # Test i64 splat
    dense_i64_splat =
      -10_000_000_000
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([2], Type.i64(opts)), opts)

    assert Enum.count(dense_i64_splat) == 2
    assert Enum.member?(dense_i64_splat, -10_000_000_000)

    assert MLIR.CAPI.mlirDenseElementsAttrGetInt64Value(dense_i64_splat, 0)
           |> Beaver.Native.to_term() == -10_000_000_000

    # Test u64 splat
    dense_u64_splat =
      18_000_000_000
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([2], Type.ui64(opts)), opts)

    assert Enum.count(dense_u64_splat) == 2
    assert Enum.member?(dense_u64_splat, 18_000_000_000)

    assert MLIR.CAPI.mlirDenseElementsAttrGetUInt64Value(dense_u64_splat, 0)
           |> Beaver.Native.to_term() == 18_000_000_000

    # Test f32 splat
    dense_f32_splat =
      0.5
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.f32(opts)), opts)

    assert Enum.count(dense_f32_splat) == 3
    assert Enum.member?(dense_f32_splat, 0.5)

    assert MLIR.CAPI.mlirDenseElementsAttrGetFloatValue(dense_f32_splat, 0)
           |> Beaver.Native.to_term() == 0.5

    # Test f64 splat
    dense_f64_splat =
      2.71828
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor!([3], Type.f64(opts)), opts)

    assert Enum.count(dense_f64_splat) == 3
    assert Enum.member?(dense_f64_splat, 2.71828)

    assert MLIR.CAPI.mlirDenseElementsAttrGetDoubleValue(dense_f64_splat, 0)
           |> Beaver.Native.to_term() == 2.71828
  end

  test "dictionaries", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Create test dictionary
    dict =
      [
        {"key1", MLIR.Attribute.integer(Type.i32(opts), 1)},
        {"key2", MLIR.Attribute.integer(Type.i32(opts), 2)}
      ]
      |> Enum.map(fn {k, v} -> MLIR.NamedAttribute.get(k, v, opts) end)
      |> MLIR.Attribute.dictionary(opts)

    # Test Enum protocol functions
    assert Enum.count(dict) == 2

    assert Enum.member?(
             dict,
             MLIR.NamedAttribute.get("key1", MLIR.Attribute.integer(Type.i32(opts), 1), opts)
           )

    refute Enum.member?(
             dict,
             MLIR.NamedAttribute.get("key3", MLIR.Attribute.integer(Type.i32(opts), 3), opts)
           )

    # Test Enumerable protocol functions
    assert Enumerable.count(dict) == {:ok, 2}

    assert Enumerable.member?(
             dict,
             MLIR.NamedAttribute.get("key2", MLIR.Attribute.integer(Type.i32(opts), 2), opts)
           ) == {:ok, true}

    # Test reduce operations
    sum =
      Enum.reduce(dict, 0, fn named_attr, acc ->
        attr = MLIR.NamedAttribute.attribute(named_attr)
        acc + Beaver.Native.to_term(MLIR.CAPI.mlirIntegerAttrGetValueInt(attr))
      end)

    assert sum == 3

    # Test slice operations
    assert Enum.slice(dict, 0..1) |> Enum.count() == 2
    assert {:ok, 2, _} = Enumerable.slice(dict)
  end
end
