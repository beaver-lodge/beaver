defmodule AttributeAccessTest do
  @moduledoc "Test the Access behavior implementation for MLIR attributes."
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias MLIR.Type

  test "arrays", %{ctx: ctx} do
    integer = &MLIR.Attribute.integer(Type.i32(ctx: ctx), &1)
    array = &MLIR.Attribute.array(Enum.map(&1, integer), ctx: ctx)
    arr = array.(0..5)
    for i <- [-100, 100], do: assert(is_nil(arr[i]))

    for {i, value} <- [{0, 0}, {5, 5}, {-1, 5}, {-2, 4}], do: assert(MLIR.equal?(integer.(value), arr[i]))

    {v, remaining} = Access.pop(arr, 3)
    assert MLIR.equal?(integer.(3), v)
    assert MLIR.equal?(array.([0, 1, 2, 4, 5]), remaining)
    refute MLIR.equal?(remaining, arr)
    assert MLIR.equal?(put_in(arr[2], integer.(20)), array.([0, 1, 20, 3, 4, 5]))
    assert MLIR.equal?(arr, put_in(arr[200], :xx))
    assert {nil, unchanged} = Access.pop(arr, 30)
    assert MLIR.equal?(arr, unchanged)
  end

  for {native, values, index} <- [
        {Beaver.Native.I32, [0, 1, 2, 3, 4, 5], 3},
        {Beaver.Native.F32, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 1}
      ] do
    test "dense array #{native}", %{ctx: ctx} do
      values = unquote(values)
      attr = MLIR.Attribute.dense_array(values, unquote(native), ctx: ctx)
      for {v, i} <- Enum.with_index(values), do: assert(attr[i] == v)
      {old, updated} = Access.get_and_update(attr, 2, &{&1, &1 * 10})
      assert old == Enum.at(values, 2)
      assert updated[2] == old * 10

      for pop <- [&Access.pop(&1, unquote(index)), &Access.get_and_update(&1, unquote(index), fn _ -> :pop end)] do
        {value, rest} = pop.(attr)
        assert value == Enum.at(values, unquote(index))
        assert Enum.count(rest) == 5
      end

      assert {nil, %MLIR.Attribute{}} = Access.get_and_update(attr, 100, &{&1, &1})
    end
  end

  test "dictionaries", %{ctx: ctx} do
    integer = &MLIR.Attribute.integer(Type.i32(ctx: ctx), &1)

    dictionary = fn pairs ->
      MLIR.Attribute.dictionary(
        Enum.map(pairs, fn {k, v} -> MLIR.NamedAttribute.get(k, integer.(v), ctx: ctx) end),
        ctx: ctx
      )
    end

    dict = dictionary.([{"key1", 1}, {"key2", 2}])
    for key <- ["key1", :key1], do: assert(MLIR.equal?(integer.(1), dict[key]))
    assert {:ok, named} = Access.fetch(dict, 0)
    assert MLIR.to_string(MLIR.NamedAttribute.name(named)) == "key1"
    {popped, remaining} = Access.pop(dict, 0)
    assert Enum.count(remaining) == 1
    assert MLIR.to_string(MLIR.NamedAttribute.name(popped)) == "key1"
    assert :error = Access.fetch(dict, 100)
    {popped, _} = Access.get_and_update(dict, 0, &{&1, &1})
    assert MLIR.to_string(MLIR.NamedAttribute.name(popped)) == "key1"

    for pairs <- [[{"string_key", 1}, {:atom_key, 2}], [{"mixed_key1", 100}, {:mixed_key2, 200}]] do
      dict = dictionary.(pairs)

      for {key, value} <- pairs, k <- [to_string(key), String.to_atom(to_string(key))] do
        assert MLIR.equal?(integer.(value), dict[k])
        assert {:ok, attr} = Access.fetch(dict, k)
        assert MLIR.equal?(integer.(value), attr)
        {old, updated} = Access.get_and_update(dict, k, &{&1, integer.(value * 10)})
        assert MLIR.equal?(integer.(value), MLIR.NamedAttribute.attribute(old))
        assert MLIR.equal?(integer.(value * 10), updated[k])
      end

      empty =
        Enum.reduce(pairs, dict, fn {k, v}, acc ->
          {old, remaining} = Access.pop(acc, k)
          assert MLIR.equal?(integer.(v), MLIR.NamedAttribute.attribute(old))
          assert Enum.count(remaining) == Enum.count(acc) - 1
          remaining
        end)

      assert Enum.empty?(empty)

      for key <- ["nonexistent", :nonexistent] do
        assert is_nil(dict[key])
        assert :error = Access.fetch(dict, key)
      end
    end
  end

  for {type, values, index, replacement} <- [
        {:i32, [0, 1, 2, 3, 4, 5], 2, 20},
        {:f32, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 1, 10.0},
        {:string, ["hello", "world", "!"], 1, "updated"},
        {:i1, [true, false, true], 1, true},
        {:i8, [-1, 0, 1], 1, 10},
        {:ui8, [0, 128, 255], 1, 100},
        {:i64, [-10_000_000_000, 0, 10_000_000_000], 1, 10},
        {:f64, [1.0, 2.0, 3.0], 2, 30.0},
        {:string, [], 0, nil}
      ] do
    @tag values: values
    test "dense #{type} access #{length(values)} elements", %{ctx: ctx, values: values} do
      MLIR.Context.allow_unregistered_dialects(ctx)

      type =
        if unquote(type) == :string,
          do: Type.opaque("test", "str", ctx: ctx),
          else: apply(Type, unquote(type), [[ctx: ctx]])

      tensor = Type.ranked_tensor([length(values)], type)
      attr = MLIR.Attribute.dense_elements(values, tensor, ctx: ctx)
      for {v, i} <- Enum.with_index(values), do: assert(attr[i] == v)
      {old, updated} = Access.get_and_update(attr, unquote(index), &{&1, unquote(replacement)})
      assert old == Enum.at(values, unquote(index))
      assert updated[unquote(index)] == unquote(replacement)
      assert {nil, %MLIR.Attribute{}} = Access.get_and_update(attr, 100, &{&1, &1})

      unless values == [] do
        assert_raise ArgumentError, "cannot pop from dense elements attribute", fn -> Access.pop(attr, 0) end
      end
    end
  end

  test "access wrong attribute", %{ctx: ctx} do
    assert_raise ArgumentError, "not a container attribute", fn ->
      MLIR.Attribute.integer(Type.i(32, ctx: ctx), 0)[0]
    end
  end

  test "access layout attr", %{ctx: ctx} do
    strided_layout = MLIR.Attribute.strided_layout(10, [1, 2, 3], ctx: ctx)
    assert 1 = strided_layout[0]
    assert 2 = strided_layout[1]
    assert 3 = strided_layout[2]
    assert 10 = strided_layout[:offset]

    {get, updated} =
      Access.get_and_update(strided_layout, 2, fn val ->
        new_val = 30
        {val, new_val}
      end)

    assert 3 = get
    assert 30 = updated[2]
    assert 10 = updated[:offset]
    assert :dynamic = MLIR.Attribute.strided_layout(:dynamic, [1, 2, 3], ctx: ctx)[:offset]

    strided_layout =
      MLIR.Attribute.strided_layout(MLIR.ShapedType.dynamic_stride_or_offset(), [1, :dynamic, 3], ctx: ctx)

    assert :dynamic = strided_layout[:offset]
    assert :dynamic = strided_layout[1]
  end
end
