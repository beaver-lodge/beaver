defmodule Beaver.MLIR.CAPI.FizzTest do
  use ExUnit.Case
  alias Beaver.MLIR.CAPI

  test "bool" do
    assert CAPI.bool(true) |> CAPI.to_term()
    assert not (CAPI.bool(false) |> CAPI.to_term())
  end

  test "array i64" do
    %CAPI.Array{ref: ref, zig_t: "[*c]const i64"} = CAPI.I64.array([1, 2, 3])
    assert is_reference(ref)
  end

  test "array i32" do
    %CAPI.Array{ref: ref, zig_t: "[*c]const i32"} = CAPI.I32.array([1, 2, 3])
    assert is_reference(ref)
  end

  test "array f64" do
    %CAPI.Array{ref: ref, zig_t: "[*c]const f64"} = CAPI.F64.array([1.0, 2.0, 3.0])
    assert is_reference(ref)
  end

  test "empty array f64" do
    %CAPI.Array{ref: ref, zig_t: "[*c]const f64"} = CAPI.F64.array([])
    assert is_reference(ref)
  end

  test "empty array mlir type" do
    %Beaver.MLIR.CAPI.Array{ref: ref, zig_t: "[*c]const c.struct_MlirType"} =
      CAPI.MlirType.array([])

    assert is_reference(ref)
  end
end
