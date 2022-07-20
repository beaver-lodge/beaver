defmodule Beaver.MLIR.CAPI.FizzTest do
  use ExUnit.Case
  alias Beaver.MLIR.CAPI

  test "bool" do
    assert CAPI.bool(true) |> CAPI.to_term()
    assert not (CAPI.bool(false) |> CAPI.to_term())
  end

  test "array i64" do
    ref = CAPI.fizz_nif_get_resource_array_resource_type_i64([1, 2, 3])
    assert is_reference(ref)
  end

  test "array i32" do
    ref = CAPI.fizz_nif_get_resource_array_resource_type_i32([1, 2, 3])
    assert is_reference(ref)
  end

  test "array f64" do
    ref = CAPI.fizz_nif_get_resource_array_resource_type_f64([1.0, 2.0, 3.0])
    assert is_reference(ref)
  end

  test "empty array f64" do
    ref = CAPI.fizz_nif_get_resource_array_resource_type_f64([])
    assert is_reference(ref)
  end

  test "empty array mlir type" do
    %Beaver.MLIR.CAPI.ArrayMlirType{ref: ref, zig_t: "[*c]const c.struct_MlirType"} =
      CAPI.ArrayMlirType.create([])

    assert is_reference(ref)
  end
end
