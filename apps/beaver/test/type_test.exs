defmodule TypeTest do
  use ExUnit.Case, async: true
  alias Beaver.MLIR
  alias MLIR.Type
  import MLIR.Sigils

  test "generated type apis" do
    assert Type.equal?(Type.f16(), Type.get("f16"))
    assert Type.equal?(Type.f(16), Type.get("f16"))
    assert Type.equal?(Type.f32(), Type.get("f32"))
    assert Type.equal?(Type.f(32), Type.get("f32"))
    assert Type.equal?(Type.f64(), Type.get("f64"))
    assert Type.equal?(Type.f(64), Type.get("f64"))
    assert Type.equal?(Type.integer(1), Type.get("i1"))
    assert Type.equal?(Type.integer(16), Type.get("i16"))
    assert Type.equal?(Type.integer(32), Type.get("i32"))
    assert Type.equal?(Type.integer(32), ~t{i32})
    assert Type.equal?(Type.integer(64), Type.get("i64"))
    assert Type.equal?(Type.integer(128), Type.get("i128"))
    assert Type.equal?(Type.complex(Type.f32()), Type.get("complex<f32>"))

    assert Type.unranked_tensor(Type.complex(Type.f32())) |> Type.to_string() ==
             "tensor<*xcomplex<f32>>"

    assert Type.equal?(Type.unranked_tensor(Type.f32()), ~t{tensor<*xf32>})
  end
end
