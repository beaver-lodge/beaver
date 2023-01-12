defmodule Beaver.MLIR.CAPI.KindaTest do
  use ExUnit.Case
  alias Beaver.Native

  @moduletag :smoke
  test "bool" do
    assert Native.Bool.make(true) |> Native.to_term()
    assert not (Native.Bool.make(false) |> Native.to_term())
  end

  test "array i64" do
    %Native.Array{ref: ref} = Native.array([1, 2, 3], Native.I64)
    assert is_reference(ref)
  end

  test "array i32" do
    %Native.Array{ref: ref} = Native.array([1, 2, 3], Native.I32)
    assert is_reference(ref)
  end

  test "array f64" do
    %Native.Array{ref: ref} = Native.array([1.0, 2.0, 3.0], Native.F64)

    assert is_reference(ref)
  end

  test "empty array f64" do
    %Native.Array{ref: ref} = Native.array([], Native.F64)
    assert is_reference(ref)
  end

  test "empty array mlir type" do
    %Native.Array{ref: ref} = Native.array([], Beaver.MLIR.Type)

    assert is_reference(ref)
  end
end
