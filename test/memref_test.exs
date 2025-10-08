defmodule MemRefTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR.Dialect.MemRef
  @moduletag :smoke

  test "dynamic stride or offset" do
    assert Beaver.MLIR.ShapedType.dynamic_stride_or_offset?(
             Beaver.MLIR.ShapedType.dynamic_stride_or_offset()
           )

    refute Beaver.MLIR.ShapedType.static_stride_or_offset?(
             Beaver.MLIR.ShapedType.dynamic_stride_or_offset()
           )

    refute Beaver.MLIR.ShapedType.dynamic_stride_or_offset?(1)
    assert Beaver.MLIR.ShapedType.static_stride_or_offset?(1)

    assert Beaver.MLIR.ShapedType.dynamic_stride_or_offset?(:dynamic)
    refute Beaver.MLIR.ShapedType.static_stride_or_offset?(:dynamic)
  end

  test "memref layout and memory space", %{ctx: ctx} do
    assert_raise ArgumentError, "only ranked memref has layout", fn ->
      ~t{memref<*xbf16>}.(ctx) |> MemRef.layout()
    end

    assert_raise ArgumentError, "only ranked memref has layout", fn ->
      MLIR.Type.index(ctx: ctx) |> MemRef.layout()
    end

    assert memref_type = ~t{memref<128xbf16>}.(ctx)
    assert MLIR.ShapedType.static_dim?(memref_type, 0)
    refute MLIR.ShapedType.dynamic_dim?(memref_type, 0)
    assert 128 = MLIR.ShapedType.dim_size(memref_type, 0)

    assert MemRef.memory_space(memref_type) == nil

    assert layout = MemRef.layout(memref_type)
    assert to_string(layout) == "affine_map<(d0) -> (d0)>"

    memref_type = ~t{memref<128x?xbf16>}.(ctx)
    assert :dynamic = MLIR.ShapedType.dim_size(memref_type, 1)
    assert 2 = MLIR.ShapedType.rank(memref_type)
    refute MLIR.ShapedType.static?(memref_type)
    assert MLIR.equal?(MLIR.ShapedType.element_type(memref_type), MLIR.Type.bf16(ctx: ctx))

    assert_raise ArgumentError, "not a shaped type", fn ->
      MLIR.Type.index(ctx: ctx) |> MLIR.ShapedType.element_type()
    end
  end

  test "get strides and offset of a memref type", %{ctx: ctx} do
    type = ~t{memref<1x2xi32, strided<[?, 1], offset: 0>>}.(ctx)
    refute MLIR.null?(type)
    assert {[:dynamic, 1], 0} = MemRef.strides_and_offset(type)

    assert_raise ArgumentError, "only ranked memref has strides and offset", fn ->
      MemRef.strides_and_offset(MLIR.Type.index(ctx: ctx))
    end
  end

  test "affine map of a memref type", %{ctx: ctx} do
    type = ~t{memref<1x2xi32, affine_map<(d0, d1) -> (d0, d1)>>}.(ctx)
    refute MLIR.null?(type)
    assert affine_map = MemRef.affine_map(type)
    assert to_string(affine_map) == "(d0, d1) -> (d0, d1)"
  end

  test "num of elements of a shaped type", %{ctx: ctx} do
    assert 6 = ~t{memref<2x3xf32>}.(ctx) |> MLIR.ShapedType.num_elements()
    assert 0 = ~t{memref<0x3xf32>}.(ctx) |> MLIR.ShapedType.num_elements()

    assert_raise ArgumentError, "not a shaped type", fn ->
      MLIR.ShapedType.num_elements(MLIR.Type.index(ctx: ctx))
    end

    assert_raise ArgumentError, "cannot get element count of dynamic shaped type", fn ->
      ~t{memref<*xf32>}.(ctx) |> MLIR.ShapedType.num_elements()
    end
  end

  test "memref<?xi8>", %{ctx: ctx} do
    import Beaver.Sigils
    assert memref_type = ~t{memref<?xi8>}.(ctx)
    assert 1 = MLIR.ShapedType.rank(memref_type)
    assert :dynamic = MLIR.ShapedType.dim_size(memref_type, 0)
    assert MLIR.equal?(MLIR.ShapedType.element_type(memref_type), MLIR.Type.i8(ctx: ctx))
    assert {[1], 0} = MemRef.strides_and_offset(memref_type)
    assert MLIR.equal?(memref_type, ~t{memref<?xi8>}.(ctx))
    memref_type_from_api = MLIR.Type.memref!([:dynamic], MLIR.Type.i8(ctx: ctx))
    assert MLIR.equal?(memref_type, memref_type_from_api)

    assert memref_type_strided = ~t{memref<?xi8, strided<[1]>>}.(ctx)
    assert {[1], 0} = MemRef.strides_and_offset(memref_type_strided)
    refute MLIR.equal?(memref_type_strided, memref_type)

    assert memref_type_strided_offset = ~t{memref<?xi8, strided<[1], offset: 0>>}.(ctx)
    assert {[1], 0} = MemRef.strides_and_offset(memref_type_strided_offset)
    assert MLIR.equal?(memref_type_strided_offset, memref_type_strided)
    refute MLIR.equal?(memref_type_strided_offset, memref_type)

    memref_type_strided_offset_from_api =
      MLIR.Type.memref!([:dynamic], MLIR.Type.i8(ctx: ctx),
        layout: MLIR.Attribute.strided_layout(0, [1], ctx: ctx)
      )

    assert MLIR.equal?(memref_type_strided_offset, memref_type_strided_offset_from_api)

    memref_type_strided = ~t{memref<16xi8, strided<[2]>>}.(ctx)
    assert 16 = memref_type_strided |> MLIR.ShapedType.num_elements()
  end
end
