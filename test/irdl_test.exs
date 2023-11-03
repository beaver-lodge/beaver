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

  test "define dialect", test_context do
    use Beaver
    alias Beaver.MLIR.Dialect.Func
    require Func

    MLIR.CAPI.mlirContextSetAllowUnregisteredDialects(test_context[:ctx], true)
    MLIR.Context.load_dialect(test_context[:ctx], CMath)

    mlir ctx: test_context[:ctx] do
      module do
        Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
          region do
            block bb_entry() do
              v0 = CMath.norm(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
              CMath.mul(v0) >>> Type.i(1)
            end
          end
        end
      end
    end
    |> MLIR.dump!()
    |> MLIR.Operation.verify!()
  end
end
