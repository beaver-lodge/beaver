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
      assert Type.equal?(Type.i1(), Type.i(1))
      assert Type.equal?(Type.i8(), Type.i(8))
      assert Type.equal?(Type.i32(), Type.i(32))
      assert Type.equal?(Type.i64(), Type.i(64))
      assert Type.equal?(Type.i128(), Type.i(128))
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
      assert Attribute.equal?(Attribute.float(Type.f(32), 0.0), ~a{0.0}f32)
      assert Attribute.equal?(Attribute.integer(Type.index(), 1), ~a{1}index)

      assert not Attribute.is_null(
               Attribute.type(
                 Type.function(
                   [Type.i(32)],
                   [Type.i(32)]
                 )
               )
             )

      assert Type.function(
               [Type.ranked_tensor([1, 2, 3, 4], Type.i(32))],
               [Type.i(32)]
             )
             |> Type.to_string() ==
               "(tensor<1x2x3x4xi32>) -> i32"

      assert Attribute.equal?(
               Attribute.type(
                 Type.function(
                   [Type.i(32)],
                   [Type.i(32)]
                 )
               ),
               ~a{(i32) -> (i32)}
             )

      vec2xi32 = Type.vector([2], Type.i(32))
      assert Type.to_string(vec2xi32) == "vector<2xi32>"
      i0attr = Attribute.integer(Type.i(32), 0)

      assert Attribute.equal?(
               Attribute.dense_elements([i0attr, i0attr], vec2xi32),
               ~a{dense<0> : vector<2xi32>}
             )

      assert Attribute.equal?(
               Attribute.dense_elements([i0attr], vec2xi32),
               ~a{dense<0> : vector<2xi32>}
             )

      assert Attribute.equal?(
               MLIR.ODS.operand_segment_sizes([0, 0]),
               ~a{dense<0> : vector<2xi32>}
             )

      assert Attribute.equal?(
               MLIR.ODS.operand_segment_sizes([1, 0]),
               ~a{dense<[1, 0]> : vector<2xi32>}
             )
    end

    test "iterator_types" do
      parallel = Attribute.string("parallel")
      parallel2 = Attribute.array([parallel, parallel])

      assert Attribute.equal?(
               parallel2,
               ~a{["parallel", "parallel"]}
             )
    end
  end
end
