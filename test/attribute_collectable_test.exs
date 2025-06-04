defmodule AttributeCollectableTest do
  @moduledoc """
  Test the Collectable protocol implementation for MLIR attributes.
  """
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias MLIR.Type

  test "arrays", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test collecting into empty array
    empty = MLIR.Attribute.array([], opts)
    collected = Enum.into([1, 2, 3], empty, &MLIR.Attribute.integer(Type.i32(opts), &1))

    expected =
      [1, 2, 3]
      |> Enum.map(&MLIR.Attribute.integer(Type.i32(opts), &1))
      |> MLIR.Attribute.array(opts)

    assert MLIR.equal?(collected, expected)

    # Test collecting into existing array
    existing =
      [4, 5]
      |> Enum.map(&MLIR.Attribute.integer(Type.i32(opts), &1))
      |> MLIR.Attribute.array(opts)

    assert_raise ArgumentError, "cannot collect into an attribute that is not empty", fn ->
      Enum.into([1, 2, 3], existing, &MLIR.Attribute.integer(Type.i32(opts), &1))
    end
  end

  test "dense arrays", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test collecting into empty dense i32 array
    empty_i32 = MLIR.Attribute.dense_array([], Beaver.Native.I32, opts)
    collected_i32 = Enum.into([1, 2, 3], empty_i32, & &1)

    expected_i32 =
      [1, 2, 3]
      |> MLIR.Attribute.dense_array(Beaver.Native.I32, opts)

    assert MLIR.equal?(collected_i32, expected_i32)
    assert Enum.count(collected_i32) == 3
    assert Enum.at(collected_i32, 0) == 1
    assert Enum.at(collected_i32, 1) == 2
    assert Enum.at(collected_i32, 2) == 3

    # Test collecting into empty dense f32 array
    empty_f32 = MLIR.Attribute.dense_array([], Beaver.Native.F32, opts)
    collected_f32 = Enum.into([1.0, 2.0, 3.0], empty_f32, & &1)

    expected_f32 =
      [1.0, 2.0, 3.0]
      |> MLIR.Attribute.dense_array(Beaver.Native.F32, opts)

    assert MLIR.equal?(collected_f32, expected_f32)
    assert Enum.count(collected_f32) == 3
    assert Enum.at(collected_f32, 0) == 1.0
    assert Enum.at(collected_f32, 1) == 2.0
    assert Enum.at(collected_f32, 2) == 3.0

    # Test collecting into empty dense bool array
    empty_bool = MLIR.Attribute.dense_array([], Beaver.Native.Bool, opts)
    collected_bool = Enum.into([true, false, true], empty_bool, & &1)

    expected_bool =
      [true, false, true]
      |> MLIR.Attribute.dense_array(Beaver.Native.Bool, opts)

    assert MLIR.equal?(collected_bool, expected_bool)
    assert Enum.count(collected_bool) == 3
    assert Enum.at(collected_bool, 0) == true
    assert Enum.at(collected_bool, 1) == false
    assert Enum.at(collected_bool, 2) == true

    # Test error cases
    non_empty = MLIR.Attribute.dense_array([1], Beaver.Native.I32, opts)

    assert_raise ArgumentError, "cannot collect into an attribute that is not empty", fn ->
      Enum.into([2, 3], non_empty, & &1)
    end
  end

  test "dictionaries", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test collecting into empty dictionary
    empty_dict = MLIR.Attribute.dictionary([], opts)

    created =
      Enum.into([{"key", 1}], empty_dict, fn {k, v} ->
        MLIR.NamedAttribute.get(k, MLIR.Attribute.integer(Type.i32(opts), v), opts)
      end)

    assert Enum.count(created) == 1
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 1), created["key"])
  end

  test "dense elements", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test collecting into empty dense elements of i32
    shaped_type = MLIR.Type.ranked_tensor!([3], MLIR.Type.i32(opts))
    empty_dense = MLIR.Attribute.dense_elements([-1, -1, -1], shaped_type, opts)
    collected_i32 = Enum.into([1, 2, 3], empty_dense, & &1)

    expected_i32 =
      [1, 2, 3]
      |> MLIR.Attribute.dense_elements(shaped_type, opts)

    assert MLIR.equal?(collected_i32, expected_i32)
    assert Enum.count(collected_i32) == 3
    assert Enum.at(collected_i32, 0) == 1
    assert Enum.at(collected_i32, 1) == 2
    assert Enum.at(collected_i32, 2) == 3

    # Test collecting into empty dense elements of f32
    shaped_type = MLIR.Type.ranked_tensor!([3], MLIR.Type.f32(opts))
    empty_dense = MLIR.Attribute.dense_elements([0.1, 0.2, 0.3], shaped_type, opts)
    collected_f32 = Enum.into([1.0, 2.0, 3.0], empty_dense, & &1)

    expected_f32 =
      [1.0, 2.0, 3.0]
      |> MLIR.Attribute.dense_elements(shaped_type, opts)

    assert MLIR.equal?(collected_f32, expected_f32)
    assert Enum.count(collected_f32) == 3
    assert Enum.at(collected_f32, 0) == 1.0
    assert Enum.at(collected_f32, 1) == 2.0
    assert Enum.at(collected_f32, 2) == 3.0
  end
end
