defmodule IRDLTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen irdl", %{ctx: ctx} do
    import Beaver.Sigils

    m =
      ~m"""
      irdl.dialect @cmath {
        irdl.type @complex {
          %0 = irdl.is f32
          %1 = irdl.is f64
          %2 = irdl.any_of(%0, %1)
          irdl.parameters(elem: %2)
        }
        irdl.operation @norm {
          %0 = irdl.any
          %1 = irdl.parametric @cmath::@complex<%0>
          irdl.operands(complex: %1)
          irdl.results(norm: %0)
        }
         irdl.operation @mul {
          %0 = irdl.is f32
          %1 = irdl.is f64
          %2 = irdl.any_of(%0, %1)
          %3 = irdl.parametric @cmath::@complex<%2>
          irdl.operands(lhs: %3, rhs: %3)
          irdl.results(res: %3)
        }
      }
      """.(ctx)
      |> MLIR.verify!()

    assert m
           |> Beaver.MLIR.CAPI.mlirLoadIRDLDialects()
           |> Beaver.MLIR.LogicalResult.success?()

    found = MapSet.new(MLIR.Dialect.Registry.ops("cmath", ctx: ctx))
    expected = MapSet.new(["mul", "norm"])
    assert MapSet.equal?(found, expected)
  end

  test "cmath dialect", %{ctx: ctx} do
    use Beaver
    alias Beaver.MLIR.Dialect.Func
    alias Beaver.MLIR.Type
    require Func

    CMath.__slang_dialect__(ctx) |> MLIR.verify!()
    Beaver.Slang.load(ctx, CMath)

    CMath.IRExample.get(ctx)
    CMath.IRExample.gen(ctx)

    assert not (CMath.some_attr(Type.f32())
                |> Beaver.Deferred.create(ctx)
                |> MLIR.null?())

    assert not (MLIR.CAPI.mlirContextGetOrLoadDialect(
                  ctx,
                  MLIR.StringRef.create("cmath")
                )
                |> MLIR.null?())
  end

  test "var dialect", %{ctx: ctx} do
    use Beaver
    TestVariadic.__slang_dialect__(ctx) |> MLIR.verify!()
  end
end
