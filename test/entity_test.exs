defmodule EntityTest do
  @moduledoc """
  Test the creation of MLIR entities including attributes, types and locations.
  """
  use Beaver.Case, async: true, diagnostic: :server
  alias Beaver.MLIR
  alias MLIR.{Type, Attribute}
  import Beaver.Sigils
  doctest Beaver
  doctest Beaver.Sigils
  doctest Beaver.MLIR.Type
  doctest Beaver.MLIR.Location

  describe "type apis" do
    test "generated", %{ctx: ctx} do
      opts = [ctx: ctx]
      assert MLIR.equal?(Type.f16(opts), Type.get("f16", opts))
      assert MLIR.equal?(Type.f(16, opts), Type.get("f16", opts))
      assert MLIR.equal?(Type.f32(opts), Type.get("f32", opts))
      assert MLIR.equal?(Type.f(32, opts), Type.get("f32", opts))
      assert MLIR.equal?(Type.f64(opts), Type.get("f64", opts))
      assert MLIR.equal?(Type.f(64, opts), Type.get("f64", opts))
      assert MLIR.equal?(Type.i(1, opts), Type.get("i1", opts))
      assert MLIR.equal?(Type.i(16, opts), Type.get("i16", opts))
      assert MLIR.equal?(Type.i(32, opts), Type.get("i32", opts))
      assert MLIR.equal?(Type.i1(opts), Type.i(1, opts))
      assert MLIR.equal?(Type.i8(opts), Type.i(8, opts))
      assert MLIR.equal?(Type.i32(opts), Type.i(32, opts))
      assert MLIR.equal?(Type.i64(opts), Type.i(64, opts))
      assert MLIR.equal?(Type.i128(opts), Type.i(128, opts))
      assert MLIR.equal?(Type.integer(32, opts), ~t{i32}.(ctx))
      assert MLIR.equal?(Type.integer(64, opts), Type.get("i64").(ctx))
      assert MLIR.equal?(Type.integer(128, opts), Type.get("i128").(ctx))
      assert MLIR.equal?(Type.complex(Type.f32()).(ctx), Type.get("complex<f32>").(ctx))

      assert Type.unranked_tensor(Type.complex(Type.f32())).(ctx) |> to_string() ==
               "tensor<*xcomplex<f32>>"

      assert MLIR.equal?(Type.unranked_tensor(Type.f32()).(ctx), ~t{tensor<*xf32>}.(ctx))

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

      assert MLIR.equal?(Type.none().(ctx), Type.get("none").(ctx))
    end
  end

  describe "attr apis" do
    test "generate", %{ctx: ctx} do
      assert MLIR.equal?(Attribute.type(Type.f32()).(ctx), Attribute.type(Type.f32()).(ctx))
      assert MLIR.equal?(Attribute.type(Type.f32()), Attribute.type(Type.f32()).(ctx))
      assert MLIR.equal?(Attribute.type(Type.f32()).(ctx), Attribute.type(Type.f32()))

      assert Attribute.integer(Type.i(32), 1) |> Beaver.Deferred.create(ctx) |> to_string() ==
               "1 : i32"

      assert MLIR.equal?(Attribute.integer(Type.i(32), 0).(ctx), ~a{0}i32.(ctx))
      assert MLIR.equal?(Attribute.float(Type.f(32), 0.0).(ctx), ~a{0.0}f32.(ctx))

      assert_raise ArgumentError, "incompatible type i32", fn ->
        Attribute.float(Type.i(32), 0.0).(ctx)
      end

      assert MLIR.equal?(Attribute.integer(Type.index(), 1).(ctx), ~a{1}index.(ctx))

      assert_raise ArgumentError, "incompatible type f32", fn ->
        Attribute.integer(Type.f32(), 1).(ctx)
      end

      assert not MLIR.null?(
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

      assert MLIR.equal?(
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

      assert MLIR.equal?(
               Attribute.dense_elements([i0attr, i0attr], vec2xi32).(ctx),
               ~a{dense<0> : vector<2xi32>}.(ctx)
             )

      assert MLIR.equal?(
               Attribute.dense_elements([i0attr], vec2xi32).(ctx),
               ~a{dense<0> : vector<2xi32>}.(ctx)
             )

      assert MLIR.equal?(
               Attribute.dense_elements("abcd").(ctx),
               ~a{dense<[#{?a}, #{?b}, #{?c}, #{?d}]> : tensor<4xi8>}.(ctx)
             )

      assert MLIR.equal?(
               MLIR.ODS.operand_segment_sizes([0, 0]).(ctx),
               ~a{array<i32: 0, 0>}.(ctx)
             )

      assert MLIR.equal?(
               MLIR.ODS.operand_segment_sizes([1, 0]).(ctx),
               ~a{array<i32: 1, 0>}.(ctx)
             )
    end

    test "iterator_types", %{ctx: ctx} do
      parallel = Attribute.string("parallel")
      parallel2 = Attribute.array([parallel, parallel])

      assert MLIR.equal?(
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

      assert_raise RuntimeError, "Attribute is null", fn ->
        null |> MLIR.dump!()
      end
    end

    test "symbol name", %{ctx: ctx} do
      assert Attribute.string("foo") |> MLIR.to_string(ctx: ctx) == "\"foo\""
      assert Attribute.string(__MODULE__) |> MLIR.to_string(ctx: ctx) == "\"#{__MODULE__}\""
    end

    test "nested symbol", %{ctx: ctx} do
      ccc = MLIR.Attribute.flat_symbol_ref("ccc", ctx: ctx)
      aaa_bbb_ccc = "@aaa::@bbb::@ccc"

      assert aaa_bbb_ccc ==
               MLIR.Attribute.symbol_ref("aaa", ["bbb", "ccc"], ctx: ctx) |> MLIR.to_string()

      assert aaa_bbb_ccc ==
               MLIR.Attribute.symbol_ref("aaa", ["bbb", ccc], ctx: ctx) |> MLIR.to_string()
    end
  end

  test "identifier", %{ctx: ctx} do
    str = "foo"
    assert str == Beaver.MLIR.Identifier.get(str, ctx: ctx) |> to_string()
    a = Beaver.MLIR.Identifier.get(str, ctx: ctx)
    b = Beaver.MLIR.Identifier.get(str, ctx: ctx)
    assert MLIR.equal?(a, b)
  end

  describe "null" do
    test "attr", %{ctx: ctx, diagnostic_server: diagnostic_server} do
      assert_raise RuntimeError, "fail to parse attribute: ???", fn ->
        Attribute.get("???", ctx: ctx) |> MLIR.null?()
      end

      assert Beaver.Capturer.collect(diagnostic_server) =~
               "expected attribute value"
    end
  end
end
