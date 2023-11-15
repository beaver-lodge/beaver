defmodule IRDLTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  @moduletag :smoke

  test "gen irdl", test_context do
    import Beaver.MLIR.Sigils
    ctx = test_context[:ctx]

    m =
      ~m"""
      irdl.dialect @cmath {
        irdl.type @complex {
          %0 = irdl.is f32
          %1 = irdl.is f64
          %2 = irdl.any_of(%0, %1)
          irdl.parameters(%2)
        }
        irdl.operation @norm {
          %0 = irdl.any
          %1 = irdl.parametric @complex<%0>
          irdl.operands(%1)
          irdl.results(%0)
        }
        irdl.operation @mul {
          %0 = irdl.is f32
          %1 = irdl.is f64
          %2 = irdl.any_of(%0, %1)
          %3 = irdl.parametric @complex<%2>
          irdl.operands(%3, %3)
          irdl.results(%3)
        }
      }
      """.(ctx)
      |> MLIR.Operation.verify!()

    assert m
           |> Beaver.MLIR.CAPI.beaverLoadIRDLDialects()
           |> Beaver.MLIR.LogicalResult.success?()

    assert ["mul", "norm"] == MLIR.Dialect.Registry.ops("cmath", ctx: ctx)
  end

  test "cmath dialect",
       test_context do
    use Beaver
    alias Beaver.MLIR.Dialect.Func
    alias Beaver.MLIR.Type
    require Func

    CMath.__slang_dialect__(test_context[:ctx]) |> MLIR.Operation.verify!()
    Beaver.Slang.load(test_context[:ctx], CMath)

    CMath.IRExample.get(test_context[:ctx])
    CMath.IRExample.gen(test_context[:ctx])

    assert not (CMath.some_attr(Type.f32())
                |> Beaver.Deferred.create(test_context[:ctx])
                |> MLIR.is_null())

    assert not (MLIR.CAPI.mlirContextGetOrLoadDialect(
                  test_context[:ctx],
                  MLIR.StringRef.create("cmath")
                )
                |> MLIR.is_null())
  end

  test "var dialect",
       test_context do
    use Beaver
    TestVariadic.__slang_dialect__(test_context[:ctx]) |> MLIR.Operation.verify!()
  end
end
