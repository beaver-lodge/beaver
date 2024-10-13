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
    test "generated", test_context do
      ctx = test_context[:ctx]
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

      assert Type.unranked_tensor(Type.complex(Type.f32())).(ctx) |> to_string() ==
               "tensor<*xcomplex<f32>>"

      assert Type.equal?(Type.unranked_tensor(Type.f32()).(ctx), ~t{tensor<*xf32>}.(ctx))

      assert Type.ranked_tensor([], Type.f32()).(ctx) |> to_string() ==
               "tensor<f32>"

      assert Type.ranked_tensor([1, 2], Type.f32()).(ctx)
             |> to_string() ==
               "tensor<1x2xf32>"

      assert Type.ranked_tensor([1, :dynamic, 2], Type.f32()).(ctx)
             |> to_string() ==
               "tensor<1x?x2xf32>"

      assert Type.memref([], Type.f32()).(ctx)
             |> to_string() ==
               "memref<f32>"

      assert Type.equal?(Type.none().(ctx), Type.get("none").(ctx))
    end
  end

  describe "attr apis" do
    test "generate", test_context do
      ctx = test_context[:ctx]
      assert Attribute.equal?(Attribute.type(Type.f32()).(ctx), Attribute.type(Type.f32()).(ctx))
      assert Attribute.equal?(Attribute.type(Type.f32()), Attribute.type(Type.f32()).(ctx))
      assert Attribute.equal?(Attribute.type(Type.f32()).(ctx), Attribute.type(Type.f32()))

      assert Attribute.integer(Type.i(32), 1) |> Beaver.Deferred.create(ctx) |> to_string() ==
               "1 : i32"

      assert Attribute.equal?(Attribute.integer(Type.i(32), 0).(ctx), ~a{0}i32.(ctx))
      assert Attribute.equal?(Attribute.float(Type.f(32), 0.0).(ctx), ~a{0.0}f32.(ctx))

      assert_raise ArgumentError, "incompatible type", fn ->
        Attribute.float(Type.i(32), 0.0).(ctx)
      end

      assert Attribute.equal?(Attribute.integer(Type.index(), 1).(ctx), ~a{1}index.(ctx))

      assert_raise ArgumentError, "incompatible type", fn ->
        Attribute.integer(Type.f32(), 1).(ctx)
      end

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
             |> to_string() ==
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
               Attribute.dense_elements("abcd").(ctx),
               ~a{dense<[#{?a}, #{?b}, #{?c}, #{?d}]> : tensor<4xi8>}.(ctx)
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

    test "iterator_types", test_context do
      ctx = test_context[:ctx]
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

    test "null" do
      null = Attribute.null()

      assert_raise RuntimeError, ~r"can't dump null", fn ->
        null |> MLIR.dump!()
      end
    end

    test "symbol name", test_context do
      ctx = test_context[:ctx]
      assert Attribute.string("foo") |> MLIR.to_string(ctx: ctx) == "\"foo\""
      assert Attribute.string(__MODULE__) |> MLIR.to_string(ctx: ctx) == "\"#{__MODULE__}\""
    end

    test "nested symbol", test_context do
      ctx = test_context[:ctx]
      ccc = MLIR.Attribute.flat_symbol_ref("ccc", ctx: ctx)
      aaa_bbb_ccc = "@aaa::@bbb::@ccc"

      assert aaa_bbb_ccc ==
               MLIR.Attribute.symbol_ref("aaa", ["bbb", "ccc"], ctx: ctx) |> MLIR.to_string()

      assert aaa_bbb_ccc ==
               MLIR.Attribute.symbol_ref("aaa", ["bbb", ccc], ctx: ctx) |> MLIR.to_string()
    end
  end
end
