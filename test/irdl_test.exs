defmodule IRDLTest do
  use Beaver.Case, async: true
  @moduletag :smoke
  test "gen irdl", %{ctx: ctx} do
    m =
      sigil_m(
        "irdl.dialect @cmath {\n  irdl.type @complex {\n    %0 = irdl.is f32\n    %1 = irdl.is f64\n    %2 = irdl.any_of(%0, %1)\n    irdl.parameters(elem: %2)\n  }\n  irdl.operation @norm {\n    %0 = irdl.any\n    %1 = irdl.parametric @cmath::@complex<%0>\n    irdl.operands(complex: %1)\n    irdl.results(norm: %0)\n  }\n   irdl.operation @mul {\n    %0 = irdl.is f32\n    %1 = irdl.is f64\n    %2 = irdl.any_of(%0, %1)\n    %3 = irdl.parametric @cmath::@complex<%2>\n    irdl.operands(lhs: %3, rhs: %3)\n    irdl.results(res: %3)\n  }\n}\n",
        []
      )
      |> Beaver.Deferred.resolve(ctx)
      |> MLIR.verify!()

    assert m |> Beaver.MLIR.CAPI.mlirLoadIRDLDialects() |> Beaver.MLIR.LogicalResult.success?()
    found = MapSet.new(MLIR.Dialect.Registry.ops("cmath", ctx: ctx))
    expected = MapSet.new(["mul", "norm"])
    assert MapSet.equal?(found, expected)
  end

  test "cmath dialect", %{ctx: ctx} do
    alias Beaver.MLIR.Type
    CMath.__slang_dialect__(ctx) |> MLIR.verify!()
    Beaver.Slang.load(ctx, CMath)
    CMath.IRExample.get(ctx)
    CMath.IRExample.gen(ctx)
    assert not (CMath.some_attr(Type.f32()) |> Beaver.Deferred.resolve(ctx) |> MLIR.null?())
    assert not (MLIR.CAPI.mlirContextGetOrLoadDialect(ctx, MLIR.StringRef.create("cmath")) |> MLIR.null?())
  end

  test "var dialect", %{ctx: ctx} do
    TestVariadic.__slang_dialect__(ctx) |> MLIR.verify!()
  end

  test "region dialect", %{ctx: ctx} do
    dialect = TestRegion.__slang_dialect__(ctx) |> MLIR.verify!()
    rendered = MLIR.to_string(dialect)
    assert rendered =~ "irdl.region()"
    assert rendered =~ "irdl.region(%"
    assert rendered =~ "with size 2"
  end
end
