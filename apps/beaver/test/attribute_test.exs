defmodule AttributeTest do
  use ExUnit.Case, async: true
  alias Beaver.MLIR
  alias MLIR.{Type, Attribute}
  import MLIR.Sigils

  describe "type apis" do
    test "generated" do
      assert Type.equal?(Type.f16(), Type.get("f16"))
      assert Type.equal?(Type.f(16), Type.get("f16"))
      assert Type.equal?(Type.f32(), Type.get("f32"))
      assert Type.equal?(Type.f(32), Type.get("f32"))
      assert Type.equal?(Type.f64(), Type.get("f64"))
      assert Type.equal?(Type.f(64), Type.get("f64"))
      assert Type.equal?(Type.i(1), Type.get("i1"))
      assert Type.equal?(Type.i(16), Type.get("i16"))
      assert Type.equal?(Type.i(32), Type.get("i32"))
      assert Type.equal?(Type.integer(32), ~t{i32})
      assert Type.equal?(Type.integer(64), Type.get("i64"))
      assert Type.equal?(Type.integer(128), Type.get("i128"))
      assert Type.equal?(Type.complex(Type.f32()), Type.get("complex<f32>"))

      assert Type.unranked_tensor(Type.complex(Type.f32())) |> Type.to_string() ==
               "tensor<*xcomplex<f32>>"

      assert Type.equal?(Type.unranked_tensor(Type.f32()), ~t{tensor<*xf32>})

      assert Type.ranked_tensor([], Type.f32())
             |> Type.to_string() ==
               "tensor<f32>"

      assert Type.memref([], Type.f32())
             |> Type.to_string() ==
               "memref<f32>"
    end
  end

  describe "attr apis" do
    test "spv attributes" do
      import Beaver.MLIR.Sigils
      ~a{#spv.entry_point_abi<local_size = dense<1> : vector<3xi32>>}

      ~a{#spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>}
    end

    test "generate" do
      assert Attribute.equal?(Attribute.type(Type.f32()), Attribute.type(Type.f32()))
      assert Attribute.integer(Type.i(32), 1) |> Attribute.to_string() == "1 : i32"
      assert Attribute.equal?(Attribute.integer(Type.i(32), 0), ~a{0}i32)
    end
  end
end
