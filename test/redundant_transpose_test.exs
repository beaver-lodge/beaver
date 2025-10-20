defmodule RedundantTransposeTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  alias Beaver.MLIR.{Type}
  alias Beaver.MLIR.Dialect.{Func, TOSA}
  require Func

  test "pass to optimize redundant transpose", %{ctx: ctx} do
    use Beaver
    import Beaver.MLIR.Transform

    ir =
      mlir ctx: ctx do
        module do
          Func.func some_func(
                      function_type:
                        Type.function([TransposeHelper.tensor_t()], [TransposeHelper.tensor_t()])
                    ) do
            region do
              block _bb_entry(arg0 >>> Type.unranked_tensor(Type.f32())) do
                t =
                  TOSA.transpose(arg0, perms: TransposeHelper.perms_t_attr()) >>>
                    TransposeHelper.tensor_t()

                t =
                  TOSA.transpose(t, perms: TransposeHelper.perms_t_attr()) >>>
                    TransposeHelper.tensor_t()

                t =
                  TOSA.transpose(t, perms: TransposeHelper.perms_t_attr()) >>>
                    TransposeHelper.tensor_t()

                t =
                  TOSA.transpose(t, perms: TransposeHelper.perms_t_attr()) >>>
                    TransposeHelper.tensor_t()

                Func.return(t) >>> []
              end
            end
          end
        end
        |> MLIR.verify!()
      end

    ir_string =
      ir
      |> Beaver.Composer.nested(Func.func(), [
        DeduplicateTransposePass
      ])
      |> canonicalize
      |> Beaver.Composer.run!()
      |> MLIR.to_string()

    assert ir_string =~ "return %arg0 : tensor<*xf32>", ir_string
  end
end
