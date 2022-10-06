defmodule Beaver.MLIR.CAPI.KindaTest do
  use ExUnit.Case
  alias Beaver.MLIR.CAPI
  @moduletag :smoke
  test "bool" do
    assert Beaver.Native.Bool.make(true) |> Beaver.Native.to_term()
    assert not (Beaver.Native.Bool.make(false) |> Beaver.Native.to_term())
  end

  test "array i64" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.I64.array([1, 2, 3])
    assert is_reference(ref)
  end

  test "array i32" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.I32.array([1, 2, 3])
    assert is_reference(ref)
  end

  test "array f64" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.F64.array([1.0, 2.0, 3.0])

    assert is_reference(ref)
  end

  test "empty array f64" do
    %Beaver.Native.Array{ref: ref} = Beaver.Native.F64.array([])
    assert is_reference(ref)
  end

  test "empty array mlir type" do
    %Beaver.Native.Array{ref: ref} = CAPI.MlirType.array([])

    assert is_reference(ref)
  end
end
