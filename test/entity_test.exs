defmodule EntityTest do
  @moduledoc "Test the creation of MLIR entities including attributes, types and locations."
  use Beaver.Case, async: true
  alias Beaver.MLIR.{Attribute, FunctionType, Identifier, ShapedType, Type}
  alias Beaver.Deferred
  alias Beaver.MLIR
  import Beaver.Sigils
  doctest Beaver
  doctest Beaver.Sigils
  doctest Beaver.MLIR.Type
  doctest Beaver.MLIR.Location
  doctest Beaver.MLIR.SymbolTable

  describe "type apis" do
    test "generated", %{ctx: ctx} do
      opts = [ctx: ctx]

      for width <- [16, 32, 64] do
        parsed = Type.get("f#{width}", opts)
        assert MLIR.equal?(apply(Type, :"f#{width}", [opts]), parsed)
        assert MLIR.equal?(Type.f(width, opts), parsed)
      end

      for width <- [1, 8, 16, 32, 64, 128] do
        parsed = Type.get("i#{width}", opts)
        assert MLIR.equal?(Type.i(width, opts), parsed)
        assert MLIR.equal?(apply(Type, :"i#{width}", [opts]), parsed)
      end

      assert MLIR.equal?(Type.integer(32, opts), ~t{i32} |> Deferred.resolve(ctx))

      for width <- [64, 128],
          do: assert_same(ctx, Type.integer(width, opts), Type.get("i#{width}"))

      assert_same(ctx, Type.complex(Type.f32()), Type.get("complex<f32>"))

      assert Type.unranked_tensor!(Type.complex(Type.f32(ctx: ctx))) |> to_string() == "tensor<*xcomplex<f32>>"

      assert_same(ctx, Type.unranked_tensor!(Type.f32(ctx: ctx)), ~t{tensor<*xf32>} |> Deferred.resolve(ctx))

      for {shape, text} <- [{[], "tensor<f32>"}, {[1, 2], "tensor<1x2xf32>"}, {[1, :dynamic, 2], "tensor<1x?x2xf32>"}] do
        assert to_string(Type.ranked_tensor!(shape, Type.f32(ctx: ctx))) == text
      end

      assert to_string(Type.memref!([], Type.f32(ctx: ctx))) == "memref<f32>"
      assert_same(ctx, Type.none(), Type.get("none"))
    end
  end

  test "function type", %{ctx: ctx} do
    func_type = Type.function([Type.i32(ctx: ctx), Type.f32(ctx: ctx)], [Type.f64(ctx: ctx)]) |> Deferred.resolve(ctx)

    assert func_type |> to_string() =~ ~r"i32.+f32.+f64"
    assert 2 = FunctionType.num_inputs(func_type)
    assert MLIR.equal?(FunctionType.input(func_type, 0), Type.i32(ctx: ctx))
    assert MLIR.equal?(FunctionType.input(func_type, 1), Type.f32(ctx: ctx))
    assert 1 = FunctionType.num_results(func_type)
    assert MLIR.equal?(FunctionType.result(func_type, 0), Type.f64(ctx: ctx))
  end

  test "checked composition", %{ctx: ctx} do
    assert Type.unranked_memref!(Type.f32(ctx: ctx)) |> to_string() == "memref<*xf32>"

    assert_raise ArgumentError, "loc(unknown): invalid memref element type", fn ->
      Type.unranked_memref!(Type.none(ctx: ctx))
    end

    assert {:error, "loc(unknown): invalid memref element type"} = Type.memref([1, :dynamic, 2], Type.none(ctx: ctx))

    assert_raise ArgumentError, "calling a bang function to compose a type must be eager", fn ->
      Type.unranked_memref!(Type.i16())
    end
  end

  test "type detection", %{ctx: ctx} do
    assert Type.tensor?(~t{tensor<*xf32>} |> Deferred.resolve(ctx))
    assert Type.i(32) |> Deferred.resolve(ctx) |> Type.integer?()
    refute Type.f(32) |> Deferred.resolve(ctx) |> Type.integer?()
    assert Type.f(32) |> Deferred.resolve(ctx) |> Type.float?()
    assert Type.memref!([], Type.f32(ctx: ctx)) |> Type.memref?()
  end

  describe "attr apis" do
    test "generate", %{ctx: ctx} do
      deferred = Attribute.type(Type.f32())
      eager = Deferred.resolve(deferred, ctx)

      for {a, b} <- [{eager, eager}, {deferred, eager}, {eager, deferred}], do: assert(MLIR.equal?(a, b))

      assert Attribute.integer(Type.i(32), 1) |> Deferred.resolve(ctx) |> to_string() == "1 : i32"

      assert_same(ctx, Attribute.integer(Type.i(32), 0), ~a{0}i32)
      assert_same(ctx, Attribute.float(Type.f(32), 0.0), ~a{0.0}f32)

      assert_raise ArgumentError, "incompatible type i32", fn ->
        Attribute.float(Type.i(32), 0.0) |> Deferred.resolve(ctx)
      end

      assert_same(ctx, Attribute.integer(Type.index(), 1), ~a{1}index)

      assert_raise ArgumentError, "incompatible type f32", fn ->
        Attribute.integer(Type.f32(), 1) |> Deferred.resolve(ctx)
      end

      assert not MLIR.null?(Attribute.type(Type.function([Type.i(32)], [Type.i(32)])) |> Deferred.resolve(ctx))

      assert Type.function([Type.ranked_tensor!([1, 2, 3, 4], Type.i(32, ctx: ctx))], [Type.i(32)])
             |> Deferred.resolve(ctx)
             |> to_string() == "(tensor<1x2x3x4xi32>) -> i32"

      assert_same(ctx, Attribute.type(Type.function([Type.i(32)], [Type.i(32)])), ~a"(i32) -> (i32)")

      vec2xi32 = Type.vector!([2], Type.i(32, ctx: ctx))
      assert MLIR.to_string(vec2xi32) == "vector<2xi32>"
      i0attr = Attribute.integer(Type.i(32), 0)

      for values <- [[i0attr, i0attr], [i0attr], 0],
          do: assert_same(ctx, Attribute.dense_elements(values, vec2xi32), ~a"dense<0> : vector<2xi32>")

      assert_same(ctx, Attribute.dense_elements("abcd"), ~a{dense<[#{?a}, #{?b}, #{?c}, #{?d}]> : tensor<4xi8>})

      assert_same(
        ctx,
        MLIR.ODS.operand_segment_sizes([0, 0], ctx: ctx),
        ~a{array<i32: 0, 0>} |> Deferred.resolve(ctx)
      )

      assert_same(ctx, MLIR.ODS.operand_segment_sizes([1, 0]), ~a{array<i32: 1, 0>})

      assert_raise ArgumentError, "number of elements 0 does not match shaped type 3", fn ->
        Attribute.dense_elements([], ~t{tensor<3xi8>}, ctx: ctx)
      end
    end

    test "iterator_types", %{ctx: ctx} do
      parallel = Attribute.string("parallel")
      parallel2 = Attribute.array([parallel, parallel])
      assert_same(ctx, parallel2, ~a{["parallel", "parallel"]})
    end

    test "empty", %{ctx: ctx} do
      empty = Attribute.array([], ctx: ctx)
      assert %Attribute{} = empty
    end

    test "null" do
      null = Attribute.null()

      assert_raise RuntimeError, "Attribute is null", fn -> null |> MLIR.dump!() end
    end

    test "symbol name", %{ctx: ctx} do
      assert Attribute.string("foo") |> MLIR.to_string(ctx: ctx) == "\"foo\""
      assert Attribute.string(__MODULE__) |> MLIR.to_string(ctx: ctx) == "\"#{__MODULE__}\""
      name = ~s(string with "quotes" and /slashes/)
      assert_same(ctx, Attribute.string(name), ~a/#{name}/s)
    end

    test "nested symbol", %{ctx: ctx} do
      ccc = Attribute.flat_symbol_ref("ccc", ctx: ctx)
      aaa_bbb_ccc = "@aaa::@bbb::@ccc"

      for refs <- [["bbb", "ccc"], ["bbb", ccc]],
          do: assert(aaa_bbb_ccc == MLIR.to_string(Attribute.symbol_ref("aaa", refs, ctx: ctx)))
    end

    test "strided_layout", %{ctx: ctx} do
      strided_attr = Attribute.strided_layout(ShapedType.dynamic_stride_or_offset(), [1], ctx: ctx)

      assert not MLIR.null?(strided_attr)

      strided_attr = Attribute.strided_layout(1, [1, 2, 3]) |> Deferred.resolve(ctx)

      assert not MLIR.null?(strided_attr)
    end
  end

  test "identifier", %{ctx: ctx} do
    str = "foo"
    assert str == Identifier.get(str, ctx: ctx) |> to_string()
    a = Identifier.get(str, ctx: ctx)
    b = Identifier.get(str, ctx: ctx)
    assert MLIR.equal?(a, b)
  end

  defp assert_same(ctx, left, right),
    do: assert(MLIR.equal?(Deferred.resolve(left, ctx), Deferred.resolve(right, ctx)))
end
