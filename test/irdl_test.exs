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

  test "define dialect",
       test_context do
    use Beaver
    alias Beaver.MLIR.Dialect.Func
    require Func

    Beaver.Slang.load(test_context[:ctx], CMath)

    # original: https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/IRDL/test-cmath.mlir
    ~m"""
    func.func @conorm(%p: !cmath.complex<f32>, %q: !cmath.complex<f32>) -> f32 {
      %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
      %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
      %pq = arith.mulf %norm_p, %norm_q : f32
      return %pq : f32
    }


    func.func @conorm2(%p: !cmath.complex<f32>, %q: !cmath.complex<f32>) -> f32 {
      %pq = "cmath.mul"(%p, %q) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
      %conorm = "cmath.norm"(%pq) : (!cmath.complex<f32>) -> f32
      return %conorm : f32
    }
    """.(test_context[:ctx])
    |> MLIR.Operation.verify!()

    mlir ctx: test_context[:ctx] do
      module do
        alias Beaver.MLIR.Dialect.Arith
        cf32 = ~t{!cmath.complex<f32>}
        f32 = Type.f(32)

        Func.func conorm(function_type: Type.function([cf32, cf32], [f32])) do
          region do
            block bb_entry(p >>> cf32, q >>> cf32) do
              norm_p = CMath.norm(p) >>> f32
              norm_q = CMath.norm(q) >>> f32
              pq = Arith.mulf(norm_p, norm_q) >>> f32
              Func.return(pq) >>> []
            end
          end
        end

        Func.func conorm2(function_type: Type.function([cf32, cf32], [f32])) do
          region do
            block bb_entry(p >>> cf32, q >>> cf32) do
              pq = CMath.mul(p, q) >>> cf32
              conorm = CMath.norm(pq) >>> f32
              Func.return(conorm) >>> []
            end
          end
        end
      end
    end
    |> MLIR.Operation.verify!()
  end
end
