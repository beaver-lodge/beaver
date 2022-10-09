defmodule EntityTest do
  @moduledoc """
  Test the creation of MLIR entities including attributes, types and locations.
  """
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias MLIR.{Type, Attribute}
  import MLIR.Sigils
  doctest Beaver.MLIR.Sigils
  doctest Beaver.MLIR.Type
  doctest Beaver.MLIR.Location

  describe "type apis" do
    test "generated", context do
      ctx = context[:ctx]
      opts = [ctx: ctx]
      assert Type.equal?(Type.f16(opts), Type.get("f16", opts))
      assert Type.equal?(Type.f(16, opts), Type.get("f16", opts))
      assert Type.equal?(Type.f32(opts), Type.get("f32", opts))
      assert Type.equal?(Type.f(32, opts), Type.get("f32", opts))
      assert Type.equal?(Type.f64(opts), Type.get("f64", opts))
      assert Type.equal?(Type.f(64, opts), Type.get("f64", opts))
      assert Type.equal?(Type.i(1, opts), Type.get("i1", opts))
      assert Type.equal?(Type.i(16, opts), Type.get("i16", opts))
      assert Type.equal?(Type.i(32, opts), Type.get("i32", opts))
      assert Type.equal?(Type.i1(opts), Type.i(1, opts))
      assert Type.equal?(Type.i8(opts), Type.i(8, opts))
      assert Type.equal?(Type.i32(opts), Type.i(32, opts))
      assert Type.equal?(Type.i64(opts), Type.i(64, opts))
      assert Type.equal?(Type.i128(opts), Type.i(128, opts))
      assert Type.equal?(Type.integer(32, opts), ~t{i32}.(ctx))
      assert Type.equal?(Type.integer(64, opts), Type.get("i64").(ctx))
      assert Type.equal?(Type.integer(128, opts), Type.get("i128").(ctx))
      assert Type.equal?(Type.complex(Type.f32()).(ctx), Type.get("complex<f32>").(ctx))

      assert Type.unranked_tensor(Type.complex(Type.f32())).(ctx) |> MLIR.to_string() ==
               "tensor<*xcomplex<f32>>"

      assert Type.equal?(Type.unranked_tensor(Type.f32()).(ctx), ~t{tensor<*xf32>}.(ctx))

      assert Type.ranked_tensor([], Type.f32()).(ctx)
             |> MLIR.to_string() ==
               "tensor<f32>"

      assert Type.memref([], Type.f32()).(ctx)
             |> MLIR.to_string() ==
               "memref<f32>"
    end
  end

  describe "attr apis" do
    test "spirv attributes", context do
      ctx = context[:ctx]
      import Beaver.MLIR.Sigils
      ~a{#spirv.entry_point_abi<local_size = dense<1> : vector<3xi32>>}.(ctx)

      ~a{#spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>}.(
        ctx
      )
    end

    test "generate", context do
      ctx = context[:ctx]
      assert Attribute.equal?(Attribute.type(Type.f32()).(ctx), Attribute.type(Type.f32()).(ctx))

      assert Attribute.integer(Type.i(32), 1) |> Beaver.Deferred.create(ctx) |> MLIR.to_string() ==
               "1 : i32"

      assert Attribute.equal?(Attribute.integer(Type.i(32), 0).(ctx), ~a{0}i32.(ctx))
      assert Attribute.equal?(Attribute.float(Type.f(32), 0.0).(ctx), ~a{0.0}f32.(ctx))
      assert Attribute.equal?(Attribute.integer(Type.index(), 1).(ctx), ~a{1}index.(ctx))

      assert not Attribute.is_null(
               Attribute.type(
                 Type.function(
                   [Type.i(32)],
                   [Type.i(32)]
                 )
               ).(ctx)
             )

      assert Type.function(
               [Type.ranked_tensor([1, 2, 3, 4], Type.i(32))],
               [Type.i(32)]
             ).(ctx)
             |> MLIR.to_string() ==
               "(tensor<1x2x3x4xi32>) -> i32"

      assert Attribute.equal?(
               Attribute.type(
                 Type.function(
                   [Type.i(32)],
                   [Type.i(32)]
                 )
               ).(ctx),
               ~a{(i32) -> (i32)}.(ctx)
             )

      vec2xi32 = Type.vector([2], Type.i(32)).(ctx)
      assert MLIR.to_string(vec2xi32) == "vector<2xi32>"
      i0attr = Attribute.integer(Type.i(32), 0)

      assert Attribute.equal?(
               Attribute.dense_elements([i0attr, i0attr], vec2xi32).(ctx),
               ~a{dense<0> : vector<2xi32>}.(ctx)
             )

      assert Attribute.equal?(
               Attribute.dense_elements([i0attr], vec2xi32).(ctx),
               ~a{dense<0> : vector<2xi32>}.(ctx)
             )

      assert Attribute.equal?(
               MLIR.ODS.operand_segment_sizes([0, 0]).(ctx),
               ~a{array<i32: 0, 0>}.(ctx)
             )

      assert Attribute.equal?(
               MLIR.ODS.operand_segment_sizes([1, 0]).(ctx),
               ~a{array<i32: 1, 0>}.(ctx)
             )
    end

    test "iterator_types", context do
      ctx = context[:ctx]
      parallel = Attribute.string("parallel")
      parallel2 = Attribute.array([parallel, parallel])

      assert Attribute.equal?(
               parallel2.(ctx),
               ~a{["parallel", "parallel"]}.(ctx)
             )
    end

    test "empty" do
      empty = Attribute.array([])

      assert empty
    end
  end
end
