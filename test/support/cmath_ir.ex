defmodule CMath.IRExample do
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func

  def gen(ctx) do
    mlir ctx: ctx do
      module do
        cf32 = CMath.complex(Type.f32())
        f32 = Type.f(32)

        Func.func conorm(function_type: Type.function([cf32, cf32], [f32])) do
          region do
            block _bb_entry(p >>> cf32, q >>> cf32) do
              norm_p = CMath.norm(p) >>> f32
              norm_q = CMath.norm(q) >>> f32
              pq = Arith.mulf(norm_p, norm_q) >>> f32
              Func.return(pq) >>> []
            end
          end
        end

        Func.func conorm2(function_type: Type.function([cf32, ~t{!cmath.complex<f32>}], [f32])) do
          region do
            block _bb_entry(p >>> cf32, q >>> cf32) do
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

  def get(ctx) do
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
    """.(ctx)
    |> MLIR.Operation.verify!()
  end
end
