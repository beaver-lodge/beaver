defmodule Beaver.MLIR.CAPI.KindaTest do
  use ExUnit.Case
  alias Beaver.MLIR.CAPI
  @moduletag :smoke
  test "bool" do
    assert Beaver.Native.Bool.make(true) |> Beaver.Native.to_term()
    assert not (Beaver.Native.Bool.make(false) |> Beaver.Native.to_term())
  end

  test "array i64" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.array([1, 2, 3], Beaver.Native.I64)
    assert is_reference(ref)
  end

  test "array i32" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.array([1, 2, 3], Beaver.Native.I32)
    assert is_reference(ref)
  end

  test "array f64" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.array([1.0, 2.0, 3.0], Beaver.Native.F64)

    assert is_reference(ref)
  end

  test "empty array f64" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.array([], Beaver.Native.F64)
    assert is_reference(ref)
  end

  test "empty array mlir type" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.array([], CAPI.MlirType)

    assert is_reference(ref)
  end
end
