defmodule AttributeAccessTest do
  @moduledoc """
  Test the Access behavior implementation for MLIR attributes.
  """
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias MLIR.Type

  test "arrays", %{ctx: ctx} do
    opts = [ctx: ctx]

    arr =
      Range.new(0, 5, 1)
      |> Enum.map(&MLIR.Attribute.integer(Type.i32(opts), &1))
      |> MLIR.Attribute.array(opts)

    assert nil == arr[-100]
    assert nil == arr[100]
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 0), arr[0])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 5), arr[5])
    assert MLIR.equal?(arr[-1], arr[5])
    assert MLIR.equal?(arr[-2], arr[4])

    arr2_expected =
      [0, 1, 2, 4, 5]
      |> Enum.map(&MLIR.Attribute.integer(Type.i32(opts), &1))
      |> MLIR.Attribute.array(opts)

    arr3_expected =
      [0, 1, 20, 3, 4, 5]
      |> Enum.map(&MLIR.Attribute.integer(Type.i32(opts), &1))
      |> MLIR.Attribute.array(opts)

    {v, arr2} = Access.pop(arr, 3)
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 3), v)
    assert MLIR.equal?(arr2_expected, arr2)
    refute MLIR.equal?(arr2, arr)

    arr3 = put_in(arr[2], MLIR.Attribute.integer(Type.i32(opts), 20))
    assert MLIR.equal?(arr3, arr3_expected)
    identity_after_put = put_in(arr[200], :xx)
    assert MLIR.equal?(arr, identity_after_put)
    assert {nil, identity_after_put} = Access.pop(arr, 30)
    assert MLIR.equal?(arr, identity_after_put)
  end

  test "dense arrays", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test dense i32 array
    dense_i32 =
      [0, 1, 2, 3, 4, 5]
      |> MLIR.Attribute.dense_array(Beaver.Native.I32, opts)

    assert 0 = dense_i32[0]
    assert 5 = dense_i32[5]

    # Test get_and_update with value update
    {get, updated} =
      Access.get_and_update(dense_i32, 2, fn val ->
        new_val = val * 10
        {val, new_val}
      end)

    assert 2 = get
    assert 20 = updated[2]

    # Test get_and_update with :pop
    {popped_get, popped_updated} =
      Access.get_and_update(dense_i32, 3, fn _val ->
        :pop
      end)

    assert 3 = popped_get
    assert Enum.count(popped_updated) == 5

    # Test pop
    {popped, remaining} = Access.pop(dense_i32, 3)
    assert 3 = popped
    assert Enum.count(remaining) == 5

    # Test dense f32 array
    dense_f32 =
      [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
      |> MLIR.Attribute.dense_array(Beaver.Native.F32, opts)

    # Test get_and_update with :pop
    {get_f_pop, updated_f_pop} =
      Access.get_and_update(dense_f32, 1, fn _val ->
        :pop
      end)

    assert 1.0 = get_f_pop
    assert Enum.count(updated_f_pop) == 5

    # Test error cases
    assert {nil, %MLIR.Attribute{}} = Access.get_and_update(dense_i32, 100, fn x -> {x, x} end)
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

    # Test fetch
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 1), dict["key1"])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 1), dict[:key1])
    assert {:ok, named_attr} = Access.fetch(dict, 0)
    assert MLIR.NamedAttribute.name(named_attr) |> MLIR.to_string() == "key1"

    # Test pop
    {popped, remaining} = Access.pop(dict, 0)
    assert Enum.count(remaining) == 1
    assert MLIR.NamedAttribute.name(popped) |> MLIR.to_string() == "key1"

    assert :error = Access.fetch(dict, 100)
    {popped, _} = Access.get_and_update(dict, 0, fn x -> {x, x} end)
    assert MLIR.NamedAttribute.name(popped) |> MLIR.to_string() == "key1"

    # Test string/atom key access
    dict2 =
      [
        {"string_key", MLIR.Attribute.integer(Type.i32(opts), 1)},
        {:atom_key, MLIR.Attribute.integer(Type.i32(opts), 2)}
      ]
      |> Enum.map(fn {k, v} -> MLIR.NamedAttribute.get(k, v, opts) end)
      |> MLIR.Attribute.dictionary(opts)

    # Test access with both string and atom keys
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 1), dict2["string_key"])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 2), dict2["atom_key"])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 1), dict2[:string_key])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 2), dict2[:atom_key])

    # Test get_and_update with string key
    {get_val, updated_dict} =
      Access.get_and_update(dict2, "string_key", fn val ->
        new_val = MLIR.Attribute.integer(Type.i32(opts), 10)
        {val, new_val}
      end)

    assert MLIR.equal?(
             MLIR.Attribute.integer(Type.i32(opts), 1),
             MLIR.NamedAttribute.attribute(get_val)
           )

    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 10), updated_dict["string_key"])

    # Test get_and_update with atom key
    {get_val, updated_dict} =
      Access.get_and_update(updated_dict, :atom_key, fn val ->
        new_val = MLIR.Attribute.integer(Type.i32(opts), 20)
        {val, new_val}
      end)

    assert MLIR.equal?(
             MLIR.Attribute.integer(Type.i32(opts), 2),
             MLIR.NamedAttribute.attribute(get_val)
           )

    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 20), updated_dict[:atom_key])

    # Test pop with string key
    {popped, remaining} = Access.pop(dict2, "string_key")

    assert MLIR.equal?(
             MLIR.Attribute.integer(Type.i32(opts), 1),
             MLIR.NamedAttribute.attribute(popped)
           )

    assert Enum.count(remaining) == 1

    # Test pop with atom key
    {popped, remaining} = Access.pop(remaining, :atom_key)

    assert MLIR.equal?(
             MLIR.Attribute.integer(Type.i32(opts), 2),
             MLIR.NamedAttribute.attribute(popped)
           )

    assert Enum.empty?(remaining)

    # Additional test cases for string/atom key access
    dict3 =
      [
        {"mixed_key1", MLIR.Attribute.integer(Type.i32(opts), 100)},
        {:mixed_key2, MLIR.Attribute.integer(Type.i32(opts), 200)}
      ]
      |> Enum.map(fn {k, v} -> MLIR.NamedAttribute.get(k, v, opts) end)
      |> MLIR.Attribute.dictionary(opts)

    # Test string access to atom key and vice versa
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 100), dict3["mixed_key1"])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 200), dict3["mixed_key2"])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 100), dict3[:mixed_key1])
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 200), dict3[:mixed_key2])

    # Test non-existent keys
    assert nil == dict3["nonexistent"]
    assert nil == dict3[:nonexistent]

    # Test fetch with string/atom keys
    assert {:ok, attr} = Access.fetch(dict3, "mixed_key1")
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 100), attr)

    assert {:ok, attr} = Access.fetch(dict3, :mixed_key2)
    assert MLIR.equal?(MLIR.Attribute.integer(Type.i32(opts), 200), attr)

    assert :error = Access.fetch(dict3, "nonexistent")
    assert :error = Access.fetch(dict3, :nonexistent)
  end

  test "dense elements", %{ctx: ctx} do
    opts = [ctx: ctx]

    # Test dense i32 elements
    dense_i32 =
      [0, 1, 2, 3, 4, 5]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor([6], Type.i(32, opts)), opts)

    assert 0 = dense_i32[0]
    assert 5 = dense_i32[5]

    # Test get_and_update
    {get, updated} =
      Access.get_and_update(dense_i32, 2, fn val ->
        new_val = val * 10
        {val, new_val}
      end)

    assert 2 = get
    assert 20 = updated[2]

    # Test pop
    assert_raise ArgumentError, "cannot pop from dense elements attribute", fn ->
      Access.pop(dense_i32, 3)
    end

    # Test dense f32 elements
    dense_f32 =
      [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor([6], Type.f(32, opts)), opts)

    {get_f, updated_f} =
      Access.get_and_update(dense_f32, 1, fn val ->
        new_val = 10.0
        {val, new_val}
      end)

    assert 1.0 = get_f
    assert 10.0 = updated_f[1]

    # Test string dense elements
    MLIR.Context.allow_unregistered_dialects(ctx)

    dense_str =
      ["hello", "world", "!"]
      |> MLIR.Attribute.dense_elements(
        Type.ranked_tensor([3], Type.opaque("test", "str", opts)),
        opts
      )

    assert "hello" = dense_str[0]
    assert "world" = dense_str[1]
    assert "!" = dense_str[2]

    # Test get_and_update with string elements
    {get_str, updated_str} =
      Access.get_and_update(dense_str, 1, fn val ->
        new_val = "updated"
        {val, new_val}
      end)

    assert "world" = get_str
    assert "updated" = updated_str[1]

    # Test pop with string elements (should raise same error)
    assert_raise ArgumentError, "cannot pop from dense elements attribute", fn ->
      Access.pop(dense_str, 0)
    end

    # Test error cases
    assert {nil, %MLIR.Attribute{}} = Access.get_and_update(dense_i32, 100, fn x -> {x, x} end)
    assert {nil, %MLIR.Attribute{}} = Access.get_and_update(dense_str, 100, fn x -> {x, x} end)

    # Test bool dense elements
    dense_bool =
      [true, false, true]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor([3], Type.i1(opts)), opts)

    assert true = dense_bool[0]
    assert false == dense_bool[1]
    assert true = dense_bool[2]

    # Test get_and_update with bool elements
    {get_bool, updated_bool} =
      Access.get_and_update(dense_bool, 1, fn val ->
        new_val = true
        {val, new_val}
      end)

    assert false == get_bool
    assert true = updated_bool[1]

    # Test i8 dense elements
    dense_i8 =
      [-1, 0, 1]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor([3], Type.i8(opts)), opts)

    assert -1 = dense_i8[0]
    assert 0 = dense_i8[1]
    assert 1 = dense_i8[2]

    # Test u8 dense elements
    dense_u8 =
      [0, 128, 255]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor([3], Type.ui8(opts)), opts)

    assert 0 = dense_u8[0]
    assert 128 = dense_u8[1]
    assert 255 = dense_u8[2]

    # Test i64 dense elements
    dense_i64 =
      [-10_000_000_000, 0, 10_000_000_000]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor([3], Type.i64(opts)), opts)

    assert -10_000_000_000 = dense_i64[0]
    assert 0 = dense_i64[1]
    assert 10_000_000_000 = dense_i64[2]

    # Test f64 dense elements
    dense_f64 =
      [1.0, 2.0, 3.0]
      |> MLIR.Attribute.dense_elements(Type.ranked_tensor([3], Type.f64(opts)), opts)

    assert 1.0 = dense_f64[0]
    assert 2.0 = dense_f64[1]
    assert 3.0 = dense_f64[2]

    # Test get_and_update with f64 elements
    {get_f64, updated_f64} =
      Access.get_and_update(dense_f64, 2, fn val ->
        new_val = 30.0
        {val, new_val}
      end)

    assert 3.0 = get_f64
    assert 30.0 = updated_f64[2]

    # Test empty dense elements
    empty_dense =
      []
      |> MLIR.Attribute.dense_elements(
        Type.ranked_tensor([0], Type.opaque("test2", "str2", opts)),
        opts
      )

    assert nil == empty_dense[0]
    assert {nil, %MLIR.Attribute{}} = Access.get_and_update(empty_dense, 0, fn x -> {x, x} end)
  end
end
